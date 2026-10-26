/***********************************************************
 *  CUDA GPU Photon Transport Kernel for MCML
 *
 *  Author: Frank Corrales Machín (@machinfc)
 *  Reference:
 *    N. López-Marín, F. Corrales-Machín, E. Castro-Alvarez,
 *    and R. Rodríguez-Fernández, "Light Transport in Biological
 *    Tissues Using GPU," Revista Cubana de Física, 33(2),
 *    122-126 (2016).
 *
 *  Physical & Architectural Design:
 *  - Solves the steady-state Radiative Transport Equation (RTE)
 *    in multi-layered turbid media via Monte Carlo photon-packet
 *    importance sampling (Hop-Drop-Spin + Fresnel/Snell interfaces).
 *  - Extends the in-kernel multi-photon persistence & fast-math ALU
 *    design (algorithms A02/A06/A09 from López-Marín & Corrales-Machín
 *    2016) from monolayer tumoral tissue (Basal Cell Carcinoma at
 *    630 nm) to arbitrary multi-layered media.
 *  - Constant-memory layer specification table (warp-broadcast reads).
 *  - Register-resident SplitMix64 + xorshift64* PRNG per GPU thread.
 *  - Persistent In-Loop Photon Respawning with Dynamic Batch Work-Queue:
 *    eliminates SIMT warp tail-latency divergence by immediately
 *    respawning a new photon packet on any lane whose photon terminates,
 *    keeping all 32 lanes per warp actively stepping in unison.
 *  - Shared-Memory Hot-Spot Sub-Grid Privatization (__shared__):
 *    caches either the entire cylindrical grid (when <= 4,096 bins) or
 *    the high-contention 64x64 subsurface pencil-beam entry zone
 *    (r -> 0, z -> 0) in 32 KB of on-chip shared memory per block,
 *    drastically reducing global L2 atomic contention on large grids.
 ****/

#include "mcml.h"
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" time_t PunchTime(char F, char *Msg);

#define MAX_GPU_LAYERS 64
#define MAX_SHARED_DOUBLES 4096 /* 32 KB per thread block (4096 * 8B) */
#define HOTSPOT_DIM 64          /* 64 x 64 = 4096 bins near (r=0, z=0) */
#define WORK_BATCH_SIZE 64ULL   /* Photons claimed per atomic queue fetch */
#define GPU_COSZERO (1.0 - 1.0E-12)
#define GPU_COS90D 1.0E-6

__constant__ LayerStruct d_layers[MAX_GPU_LAYERS];

__device__ __forceinline__ uint64_t splitmix64(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ __forceinline__ double gpu_random(uint64_t *state)
{
  uint64_t x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  uint64_t res = x * 0x2545F4914F6CDD1DULL;
  return ((double)(res >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__ double gpu_spin_theta(double g, uint64_t *rng)
{
  double cost;
  if (g == 0.0) {
    cost = 2.0 * gpu_random(rng) - 1.0;
  } else {
    double temp = (1.0 - g * g) / (1.0 - g + 2.0 * g * gpu_random(rng));
    cost = (1.0 + g * g - temp * temp) / (2.0 * g);
    if (cost < -1.0)
      cost = -1.0;
    else if (cost > 1.0)
      cost = 1.0;
  }
  return cost;
}

__device__ __forceinline__ void gpu_spin(double g, PhotonStruct *p, uint64_t *rng)
{
  double cost = gpu_spin_theta(g, rng);
  double sint = sqrt(fmax(0.0, 1.0 - cost * cost));

  double psi = 2.0 * PI * gpu_random(rng);
  double cosp = cos(psi);
  double sinp = (psi < PI) ? sqrt(fmax(0.0, 1.0 - cosp * cosp))
                           : -sqrt(fmax(0.0, 1.0 - cosp * cosp));

  double ux = p->ux;
  double uy = p->uy;
  double uz = p->uz;

  if (fabs(uz) > GPU_COSZERO) {
    p->ux = sint * cosp;
    p->uy = sint * sinp;
    p->uz = cost * SIGN(uz);
  } else {
    double temp = sqrt(fmax(1e-30, 1.0 - uz * uz));
    p->ux = sint * (ux * uz * cosp - uy * sinp) / temp + ux * cost;
    p->uy = sint * (uy * uz * cosp + ux * sinp) / temp + uy * cost;
    p->uz = -sint * cosp * temp + uz * cost;
  }
}

__device__ __forceinline__ double gpu_rfresnel(double n1, double n2, double ca1,
                                               double *ca2_ptr)
{
  double r;
  if (n1 == n2) {
    *ca2_ptr = ca1;
    r = 0.0;
  } else if (ca1 > GPU_COSZERO) {
    *ca2_ptr = ca1;
    r = (n2 - n1) / (n2 + n1);
    r *= r;
  } else if (ca1 < GPU_COS90D) {
    *ca2_ptr = 0.0;
    r = 1.0;
  } else {
    double sa1 = sqrt(fmax(0.0, 1.0 - ca1 * ca1));
    double sa2 = n1 * sa1 / n2;
    if (sa2 >= 1.0) {
      *ca2_ptr = 0.0;
      r = 1.0;
    } else {
      double ca2 = sqrt(fmax(0.0, 1.0 - sa2 * sa2));
      *ca2_ptr = ca2;
      double cap = ca1 * ca2 - sa1 * sa2;
      double cam = ca1 * ca2 + sa1 * sa2;
      double sap = sa1 * ca2 + ca1 * sa2;
      double sam = sa1 * ca2 - ca1 * sa2;
      r = 0.5 * sam * sam * (cam * cam + cap * cap) / (sap * sap * cam * cam);
    }
  }
  return r;
}

__device__ __forceinline__ void
gpu_cross_boundary(PhotonStruct *p, short layer, short num_layers, double dr,
                   double da, short nr, short na, uint64_t *rng,
                   double *target_Rd_ra, double *target_Tt_ra)
{
  if (p->uz < 0.0) {
    double uz = p->uz;
    double uz1 = 0.0;
    double ni = d_layers[layer].n;
    double nt = d_layers[layer - 1].n;
    double r = (-uz <= d_layers[layer].cos_crit0)
                   ? 1.0
                   : gpu_rfresnel(ni, nt, -uz, &uz1);
    if (gpu_random(rng) > r) {
      if (layer == 1) {
        p->uz = -uz1;
        double ird = sqrt(p->x * p->x + p->y * p->y) / dr;
        short ir = (ird > nr - 1) ? (nr - 1) : (short)ird;
        double iad = acos(-p->uz) / da;
        short ia = (iad > na - 1) ? (na - 1) : (short)iad;
        atomicAdd(&target_Rd_ra[(int)ir * (int)na + (int)ia], p->w);
        p->w = 0.0;
        p->dead = 1;
      } else {
        p->layer--;
        p->ux *= ni / nt;
        p->uy *= ni / nt;
        p->uz = -uz1;
      }
    } else {
      p->uz = -uz;
    }
  } else {
    double uz = p->uz;
    double uz1 = 0.0;
    double ni = d_layers[layer].n;
    double nt = d_layers[layer + 1].n;
    double r = (uz <= d_layers[layer].cos_crit1)
                   ? 1.0
                   : gpu_rfresnel(ni, nt, uz, &uz1);
    if (gpu_random(rng) > r) {
      if (layer == num_layers) {
        p->uz = uz1;
        double ird = sqrt(p->x * p->x + p->y * p->y) / dr;
        short ir = (ird > nr - 1) ? (nr - 1) : (short)ird;
        double iad = acos(p->uz) / da;
        short ia = (iad > na - 1) ? (na - 1) : (short)iad;
        atomicAdd(&target_Tt_ra[(int)ir * (int)na + (int)ia], p->w);
        p->w = 0.0;
        p->dead = 1;
      } else {
        p->layer++;
        p->ux *= ni / nt;
        p->uy *= ni / nt;
        p->uz = uz1;
      }
    } else {
      p->uz = -uz;
    }
  }
}

/*
 * SHARED_MODE = 1: Full grid fits in __shared__ memory (<= 4096 bins).
 * SHARED_MODE = 2: Large grid (e.g. 400x800 BCC model); caches the high-traffic
 *                  64x64 hot-spot sub-grid around (r=0, z=0) in __shared__
 *                  memory to eliminate L2 atomic contention near the beam axis.
 */
template <int SHARED_MODE>
__global__ void mcml_photon_kernel(
    unsigned long long num_photons, double rsp, double wth, double dz,
    double dr, double da, short nz, short nr, short na, short num_layers,
    uint64_t base_seed, unsigned long long *__restrict__ d_photon_counter,
    double *__restrict__ d_A_rz, double *__restrict__ d_Rd_ra,
    double *__restrict__ d_Tt_ra)
{
  extern __shared__ double s_mem[];
  int arz_size = (int)nr * (int)nz;
  int rdra_size = (int)nr * (int)na;
  int hot_nr = ((int)nr < HOTSPOT_DIM) ? (int)nr : HOTSPOT_DIM;
  int hot_nz = ((int)nz < HOTSPOT_DIM) ? (int)nz : HOTSPOT_DIM;

  double *target_A_rz = d_A_rz;
  double *target_Rd_ra = d_Rd_ra;
  double *target_Tt_ra = d_Tt_ra;

  if (SHARED_MODE == 1) {
    int total_shared = arz_size + 2 * rdra_size;
    target_A_rz = s_mem;
    target_Rd_ra = s_mem + arz_size;
    target_Tt_ra = s_mem + arz_size + rdra_size;
    for (int idx = threadIdx.x; idx < total_shared; idx += blockDim.x) {
      s_mem[idx] = 0.0;
    }
    __syncthreads();
  } else if (SHARED_MODE == 2) {
    int hot_size = hot_nr * hot_nz;
    for (int idx = threadIdx.x; idx < hot_size; idx += blockDim.x) {
      s_mem[idx] = 0.0;
    }
    __syncthreads();
  }

  long tid = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
  uint64_t rng = splitmix64(base_seed ^ (uint64_t)(tid + 1));
  if (rng == 0)
    rng = 0xCAFEBABE12345678ULL;

  PhotonStruct p;
  p.dead = 1;
  unsigned long long batch_cur = 0ULL;
  unsigned long long batch_end = 0ULL;

  /*
   * Persistent In-Loop Photon Respawning:
   * Instead of waiting for all 32 threads in a warp to finish a nested
   * while(!p.dead) loop, any lane whose photon terminates immediately claims
   * and initializes a fresh photon on the next iteration, keeping warp lanes
   * continuously active.
   */
  while (true) {
    if (p.dead) {
      if (batch_cur >= batch_end) {
        batch_cur = atomicAdd(d_photon_counter, WORK_BATCH_SIZE);
        batch_end = batch_cur + WORK_BATCH_SIZE;
        if (batch_end > num_photons)
          batch_end = num_photons;
        if (batch_cur >= num_photons)
          break;
      }
      batch_cur++;

      p.w = 1.0 - rsp;
      p.dead = 0;
      p.layer = 1;
      p.s = 0.0;
      p.sleft = 0.0;
      p.x = 0.0;
      p.y = 0.0;
      p.z = 0.0;
      p.ux = 0.0;
      p.uy = 0.0;
      p.uz = 1.0;

      if (d_layers[1].mua == 0.0 && d_layers[1].mus == 0.0) {
        p.layer = 2;
        p.z = d_layers[2].z0;
      }
    }

    short layer = p.layer;
    double mua = d_layers[layer].mua;
    double mus = d_layers[layer].mus;

    if (mua == 0.0 && mus == 0.0) {
      if (p.uz == 0.0) {
        p.dead = 1;
      } else {
        double dl_b = (p.uz > 0.0) ? (d_layers[layer].z1 - p.z) / p.uz
                                   : (d_layers[layer].z0 - p.z) / p.uz;
        p.x += dl_b * p.ux;
        p.y += dl_b * p.uy;
        p.z += dl_b * p.uz;
        gpu_cross_boundary(&p, layer, num_layers, dr, da, nr, na, &rng,
                           target_Rd_ra, target_Tt_ra);
      }
    } else {
      double mut = mua + mus;
      if (p.sleft == 0.0) {
        double rnd = gpu_random(&rng);
        p.s = -log(rnd) / mut;
      } else {
        p.s = p.sleft / mut;
        p.sleft = 0.0;
      }

      double dl_b = 0.0;
      if (p.uz > 0.0)
        dl_b = (d_layers[layer].z1 - p.z) / p.uz;
      else if (p.uz < 0.0)
        dl_b = (d_layers[layer].z0 - p.z) / p.uz;

      if (p.uz != 0.0 && p.s > dl_b) {
        p.sleft = (p.s - dl_b) * mut;
        p.s = dl_b;
        p.x += p.s * p.ux;
        p.y += p.s * p.uy;
        p.z += p.s * p.uz;
        gpu_cross_boundary(&p, layer, num_layers, dr, da, nr, na, &rng,
                           target_Rd_ra, target_Tt_ra);
      } else {
        p.x += p.s * p.ux;
        p.y += p.s * p.uy;
        p.z += p.s * p.uz;

        double izd = p.z / dz;
        short iz = (izd > nz - 1) ? (nz - 1) : (short)izd;
        double ird = sqrt(p.x * p.x + p.y * p.y) / dr;
        short ir = (ird > nr - 1) ? (nr - 1) : (short)ird;

        double dwa = p.w * mua / mut;
        p.w -= dwa;

        if (SHARED_MODE == 2 && (int)ir < hot_nr && (int)iz < hot_nz) {
          atomicAdd(&s_mem[(int)ir * hot_nz + (int)iz], dwa);
        } else {
          atomicAdd(&target_A_rz[(int)ir * (int)nz + (int)iz], dwa);
        }

        gpu_spin(d_layers[layer].g, &p, &rng);
      }
    }

    if (p.w < wth && !p.dead) {
      if (p.w == 0.0)
        p.dead = 1;
      else if (gpu_random(&rng) < CHANCE)
        p.w /= CHANCE;
      else
        p.dead = 1;
    }
  }

  if (SHARED_MODE == 1) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < arz_size; idx += blockDim.x) {
      double val = target_A_rz[idx];
      if (val > 0.0)
        atomicAdd(&d_A_rz[idx], val);
    }
    for (int idx = threadIdx.x; idx < rdra_size; idx += blockDim.x) {
      double v_rd = target_Rd_ra[idx];
      if (v_rd > 0.0)
        atomicAdd(&d_Rd_ra[idx], v_rd);
      double v_tt = target_Tt_ra[idx];
      if (v_tt > 0.0)
        atomicAdd(&d_Tt_ra[idx], v_tt);
    }
  } else if (SHARED_MODE == 2) {
    __syncthreads();
    int hot_size = hot_nr * hot_nz;
    for (int idx = threadIdx.x; idx < hot_size; idx += blockDim.x) {
      double val = s_mem[idx];
      if (val > 0.0) {
        int ir = idx / hot_nz;
        int iz = idx % hot_nz;
        atomicAdd(&d_A_rz[ir * (int)nz + iz], val);
      }
    }
  }
}

extern "C" Boolean InitDeviceCUDA(void)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return 0;
  }
  cudaFree(0);
  return 1;
}

extern "C" Boolean RunSimulationCUDA(InputStruct *In_Ptr, OutStruct *Out_Ptr)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return 0;
  }
  if (In_Ptr->num_layers + 2 > MAX_GPU_LAYERS) {
    return 0;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("[Backend] NVIDIA CUDA GPU (%s, Compute Capability %d.%d, %d SMs)\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount);

  size_t layers_bytes = (size_t)(In_Ptr->num_layers + 2) * sizeof(LayerStruct);
  if (cudaMemcpyToSymbol(d_layers, In_Ptr->layerspecs, layers_bytes) !=
      cudaSuccess) {
    return 0;
  }

  size_t arz_elems = (size_t)In_Ptr->nr * (size_t)In_Ptr->nz;
  size_t rdra_elems = (size_t)In_Ptr->nr * (size_t)In_Ptr->na;
  size_t ttra_elems = (size_t)In_Ptr->nr * (size_t)In_Ptr->na;
  size_t total_grid_elems = arz_elems + rdra_elems + ttra_elems;

  double *d_A_rz = NULL, *d_Rd_ra = NULL, *d_Tt_ra = NULL;
  unsigned long long *d_photon_counter = NULL;
  if (cudaMalloc((void **)&d_A_rz, arz_elems * sizeof(double)) != cudaSuccess)
    return 0;
  if (cudaMalloc((void **)&d_Rd_ra, rdra_elems * sizeof(double)) !=
      cudaSuccess) {
    cudaFree(d_A_rz);
    return 0;
  }
  if (cudaMalloc((void **)&d_Tt_ra, ttra_elems * sizeof(double)) !=
      cudaSuccess) {
    cudaFree(d_A_rz);
    cudaFree(d_Rd_ra);
    return 0;
  }
  if (cudaMalloc((void **)&d_photon_counter, sizeof(unsigned long long)) !=
      cudaSuccess) {
    cudaFree(d_A_rz);
    cudaFree(d_Rd_ra);
    cudaFree(d_Tt_ra);
    return 0;
  }

  cudaMemset(d_A_rz, 0, arz_elems * sizeof(double));
  cudaMemset(d_Rd_ra, 0, rdra_elems * sizeof(double));
  cudaMemset(d_Tt_ra, 0, ttra_elems * sizeof(double));
  cudaMemset(d_photon_counter, 0, sizeof(unsigned long long));

  int threads_per_block = 256;
  int blocks = prop.multiProcessorCount * 4;
  if (blocks < 30)
    blocks = 30;
  if ((long)blocks * threads_per_block > In_Ptr->num_photons) {
    blocks = (int)((In_Ptr->num_photons + threads_per_block - 1) /
                   threads_per_block);
    if (blocks < 1)
      blocks = 1;
  }

  uint64_t seed = (uint64_t)time(NULL) ^ 0x5DEECE66DULL;
  if (total_grid_elems <= MAX_SHARED_DOUBLES) {
    size_t shared_bytes = total_grid_elems * sizeof(double);
    mcml_photon_kernel<1><<<blocks, threads_per_block, shared_bytes>>>(
        (unsigned long long)In_Ptr->num_photons, Out_Ptr->Rsp, In_Ptr->Wth,
        In_Ptr->dz, In_Ptr->dr, In_Ptr->da, In_Ptr->nz, In_Ptr->nr, In_Ptr->na,
        In_Ptr->num_layers, seed, d_photon_counter, d_A_rz, d_Rd_ra, d_Tt_ra);
  } else {
    int hot_nr =
        ((int)In_Ptr->nr < HOTSPOT_DIM) ? (int)In_Ptr->nr : HOTSPOT_DIM;
    int hot_nz =
        ((int)In_Ptr->nz < HOTSPOT_DIM) ? (int)In_Ptr->nz : HOTSPOT_DIM;
    size_t shared_bytes = (size_t)(hot_nr * hot_nz) * sizeof(double);
    mcml_photon_kernel<2><<<blocks, threads_per_block, shared_bytes>>>(
        (unsigned long long)In_Ptr->num_photons, Out_Ptr->Rsp, In_Ptr->Wth,
        In_Ptr->dz, In_Ptr->dr, In_Ptr->da, In_Ptr->nz, In_Ptr->nr, In_Ptr->na,
        In_Ptr->num_layers, seed, d_photon_counter, d_A_rz, d_Rd_ra, d_Tt_ra);
  }

  if (cudaDeviceSynchronize() != cudaSuccess) {
    cudaFree(d_photon_counter);
    cudaFree(d_A_rz);
    cudaFree(d_Rd_ra);
    cudaFree(d_Tt_ra);
    return 0;
  }

  double *h_A_rz = (double *)malloc(arz_elems * sizeof(double));
  double *h_Rd_ra = (double *)malloc(rdra_elems * sizeof(double));
  double *h_Tt_ra = (double *)malloc(ttra_elems * sizeof(double));

  cudaMemcpy(h_A_rz, d_A_rz, arz_elems * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Rd_ra, d_Rd_ra, rdra_elems * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Tt_ra, d_Tt_ra, ttra_elems * sizeof(double),
             cudaMemcpyDeviceToHost);

  for (short ir = 0; ir < In_Ptr->nr; ir++) {
    for (short iz = 0; iz < In_Ptr->nz; iz++) {
      Out_Ptr->A_rz[ir][iz] += h_A_rz[(int)ir * (int)In_Ptr->nz + (int)iz];
    }
    for (short ia = 0; ia < In_Ptr->na; ia++) {
      Out_Ptr->Rd_ra[ir][ia] += h_Rd_ra[(int)ir * (int)In_Ptr->na + (int)ia];
      Out_Ptr->Tt_ra[ir][ia] += h_Tt_ra[(int)ir * (int)In_Ptr->na + (int)ia];
    }
  }

  free(h_A_rz);
  free(h_Rd_ra);
  free(h_Tt_ra);
  cudaFree(d_photon_counter);
  cudaFree(d_A_rz);
  cudaFree(d_Rd_ra);
  cudaFree(d_Tt_ra);
  return 1;
}

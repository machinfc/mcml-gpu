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
 *  - Constant-memory layer specification table with precomputed
 *    albedo, interaction reciprocals, and Henyey-Greenstein factors.
 *  - Register-resident SplitMix64 + xorshift64* PRNG per GPU thread.
 *  - Persistent In-Loop Photon Respawning with Dynamic Batch Work-Queue:
 *    eliminates SIMT warp tail-latency divergence by immediately
 *    respawning a new photon packet on any lane whose photon terminates.
 *  - Shared-Memory Hot-Spot Sub-Grid Privatization (__shared__):
 *    caches either the entire cylindrical grid (when <= 4,096 bins) or
 *    the high-contention 64x64 subsurface pencil-beam entry zone
 *    (r -> 0, z -> 0) in 32 KB of on-chip shared memory per block.
 *  - Dual Precision Modes:
 *    1. Mixed FP32/FP64 Mode (Default, --fp32): Traces photon kinematics
 *       using Turing FP32 ALUs and Special Function Units (__logf,
 *       __sincosf, __fdividef, rsqrtf) while accumulating all absorbed,
 *       reflected, and transmitted weights in strict 64-bit FP64.
 *    2. Full FP64 Mode (--fp64): End-to-end IEEE-754 double precision.
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
#define GPU_COSZERO_F (1.0f - 1.0E-6f)
#define GPU_COS90D_F 1.0E-6f
#define GPU_TWO_PI_F 6.2831853071795864769f

typedef struct {
  float z0, z1;
  float n;
  float mua, mus, mut;
  float inv_mut;
  float mua_over_mut;
  float g;
  float one_minus_g2;
  float one_plus_g2;
  float one_minus_g;
  float two_g;
  float inv_2g;
  float cos_crit0, cos_crit1;
} GpuLayerF32;

typedef struct {
  float x, y, z;
  float ux, uy, uz;
  float w;
  float s, sleft;
  short layer;
  Boolean dead;
} PhotonStructF32;

__constant__ LayerStruct d_layers[MAX_GPU_LAYERS];
__constant__ GpuLayerF32 d_layers_f32[MAX_GPU_LAYERS];

static Boolean g_use_fp64 = 0;

extern "C" void SetPrecisionCUDA(Boolean use_fp64)
{
  g_use_fp64 = use_fp64;
}

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

__device__ __forceinline__ float gpu_random_f32(uint64_t *state)
{
  uint64_t x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  uint32_t hi = (uint32_t)((x * 0x2545F4914F6CDD1DULL) >> 40);
  return ((float)hi + 0.5f) * (1.0f / 16777216.0f);
}

/* =========================================================================
 *  MIXED-PRECISION (FP32 KINEMATICS + SFU INTRINSICS / FP64 ACCUMULATORS)
 * ========================================================================= */

__device__ __forceinline__ float gpu_spin_theta_f32(const GpuLayerF32 *lyr,
                                                    uint64_t *rng)
{
  float cost;
  float rnd = gpu_random_f32(rng);
  if (lyr->g == 0.0f) {
    cost = 2.0f * rnd - 1.0f;
  } else {
    float temp =
        __fdividef(lyr->one_minus_g2, lyr->one_minus_g + lyr->two_g * rnd);
    cost = (lyr->one_plus_g2 - temp * temp) * lyr->inv_2g;
    cost = fminf(1.0f, fmaxf(-1.0f, cost));
  }
  return cost;
}

__device__ __forceinline__ void gpu_spin_f32(const GpuLayerF32 *lyr,
                                             PhotonStructF32 *p, uint64_t *rng)
{
  float cost = gpu_spin_theta_f32(lyr, rng);
  float sint = sqrtf(fmaxf(0.0f, 1.0f - cost * cost));

  float psi = GPU_TWO_PI_F * gpu_random_f32(rng);
  float sinp, cosp;
  __sincosf(psi, &sinp, &cosp);

  float ux = p->ux;
  float uy = p->uy;
  float uz = p->uz;

  if (fabsf(uz) > GPU_COSZERO_F) {
    p->ux = sint * cosp;
    p->uy = sint * sinp;
    p->uz = (uz >= 0.0f) ? cost : -cost;
  } else {
    float one_minus_uz2 = fmaxf(1e-30f, 1.0f - uz * uz);
    float inv_temp = rsqrtf(one_minus_uz2);
    float temp = one_minus_uz2 * inv_temp;
    p->ux = sint * (ux * uz * cosp - uy * sinp) * inv_temp + ux * cost;
    p->uy = sint * (uy * uz * cosp + ux * sinp) * inv_temp + uy * cost;
    p->uz = -sint * cosp * temp + uz * cost;
  }
}

__device__ __forceinline__ float gpu_rfresnel_f32(float n1, float n2, float ca1,
                                                  float *ca2_ptr)
{
  float r;
  if (n1 == n2) {
    *ca2_ptr = ca1;
    r = 0.0f;
  } else if (ca1 > GPU_COSZERO_F) {
    *ca2_ptr = ca1;
    r = __fdividef(n2 - n1, n2 + n1);
    r *= r;
  } else if (ca1 < GPU_COS90D_F) {
    *ca2_ptr = 0.0f;
    r = 1.0f;
  } else {
    float sa1 = sqrtf(fmaxf(0.0f, 1.0f - ca1 * ca1));
    float sa2 = __fdividef(n1 * sa1, n2);
    if (sa2 >= 1.0f) {
      *ca2_ptr = 0.0f;
      r = 1.0f;
    } else {
      float ca2 = sqrtf(fmaxf(0.0f, 1.0f - sa2 * sa2));
      *ca2_ptr = ca2;
      float cap = ca1 * ca2 - sa1 * sa2;
      float cam = ca1 * ca2 + sa1 * sa2;
      float sap = sa1 * ca2 + ca1 * sa2;
      float sam = sa1 * ca2 - ca1 * sa2;
      r = 0.5f * sam * sam *
          __fdividef(cam * cam + cap * cap, sap * sap * cam * cam);
    }
  }
  return r;
}

__device__ __forceinline__ void
gpu_cross_boundary_f32(PhotonStructF32 *p, short layer, short num_layers,
                       float inv_dr, float inv_da, int nr, int na,
                       uint64_t *rng, double *target_Rd_ra,
                       double *target_Tt_ra)
{
  if (p->uz < 0.0f) {
    float uz = p->uz;
    float uz1 = 0.0f;
    float ni = d_layers_f32[layer].n;
    float nt = d_layers_f32[layer - 1].n;
    float r = (-uz <= d_layers_f32[layer].cos_crit0)
                  ? 1.0f
                  : gpu_rfresnel_f32(ni, nt, -uz, &uz1);
    if (gpu_random_f32(rng) > r) {
      if (layer == 1) {
        p->uz = -uz1;
        int ir = (int)(sqrtf(p->x * p->x + p->y * p->y) * inv_dr);
        if (ir > nr - 1)
          ir = nr - 1;
        int ia = (int)(acosf(fminf(1.0f, fmaxf(0.0f, -p->uz))) * inv_da);
        if (ia > na - 1)
          ia = na - 1;
        atomicAdd(&target_Rd_ra[(size_t)ir * (size_t)na + (size_t)ia],
                  (double)p->w);
        p->w = 0.0f;
        p->dead = 1;
      } else {
        p->layer--;
        float ratio = __fdividef(ni, nt);
        p->ux *= ratio;
        p->uy *= ratio;
        p->uz = -uz1;
      }
    } else {
      p->uz = -uz;
    }
  } else {
    float uz = p->uz;
    float uz1 = 0.0f;
    float ni = d_layers_f32[layer].n;
    float nt = d_layers_f32[layer + 1].n;
    float r = (uz <= d_layers_f32[layer].cos_crit1)
                  ? 1.0f
                  : gpu_rfresnel_f32(ni, nt, uz, &uz1);
    if (gpu_random_f32(rng) > r) {
      if (layer == num_layers) {
        p->uz = uz1;
        int ir = (int)(sqrtf(p->x * p->x + p->y * p->y) * inv_dr);
        if (ir > nr - 1)
          ir = nr - 1;
        int ia = (int)(acosf(fminf(1.0f, fmaxf(0.0f, p->uz))) * inv_da);
        if (ia > na - 1)
          ia = na - 1;
        atomicAdd(&target_Tt_ra[(size_t)ir * (size_t)na + (size_t)ia],
                  (double)p->w);
        p->w = 0.0f;
        p->dead = 1;
      } else {
        p->layer++;
        float ratio = __fdividef(ni, nt);
        p->ux *= ratio;
        p->uy *= ratio;
        p->uz = uz1;
      }
    } else {
      p->uz = -uz;
    }
  }
}

template <int SHARED_MODE>
__global__ void mcml_photon_kernel_fp32(
    unsigned long long num_photons, float rsp, float wth, float inv_dz,
    float inv_dr, float inv_da, int nz, int nr, int na, short num_layers,
    uint64_t base_seed, unsigned long long *__restrict__ d_photon_counter,
    double *__restrict__ d_A_rz, double *__restrict__ d_Rd_ra,
    double *__restrict__ d_Tt_ra)
{
  extern __shared__ double s_mem[];
  int arz_size = nr * nz;
  int rdra_size = nr * na;
  int hot_nr = (nr < HOTSPOT_DIM) ? nr : HOTSPOT_DIM;
  int hot_nz = (nz < HOTSPOT_DIM) ? nz : HOTSPOT_DIM;

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

  PhotonStructF32 p;
  p.dead = 1;
  unsigned long long batch_cur = 0ULL;
  unsigned long long batch_end = 0ULL;

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

      p.w = 1.0f - rsp;
      p.dead = 0;
      p.layer = 1;
      p.s = 0.0f;
      p.sleft = 0.0f;
      p.x = 0.0f;
      p.y = 0.0f;
      p.z = 0.0f;
      p.ux = 0.0f;
      p.uy = 0.0f;
      p.uz = 1.0f;

      if (d_layers_f32[1].mut == 0.0f) {
        p.layer = 2;
        p.z = d_layers_f32[2].z0;
      }
    }

    short layer = p.layer;
    GpuLayerF32 lyr = d_layers_f32[layer];

    if (lyr.mut == 0.0f) {
      if (p.uz == 0.0f) {
        p.dead = 1;
      } else {
        float dl_b = (p.uz > 0.0f) ? __fdividef(lyr.z1 - p.z, p.uz)
                                   : __fdividef(lyr.z0 - p.z, p.uz);
        p.x += dl_b * p.ux;
        p.y += dl_b * p.uy;
        p.z += dl_b * p.uz;
        gpu_cross_boundary_f32(&p, layer, num_layers, inv_dr, inv_da, nr, na,
                               &rng, target_Rd_ra, target_Tt_ra);
      }
    } else {
      if (p.sleft == 0.0f) {
        float rnd = gpu_random_f32(&rng);
        p.s = -__logf(rnd) * lyr.inv_mut;
      } else {
        p.s = p.sleft * lyr.inv_mut;
        p.sleft = 0.0f;
      }

      float dl_b = 0.0f;
      if (p.uz > 0.0f)
        dl_b = __fdividef(lyr.z1 - p.z, p.uz);
      else if (p.uz < 0.0f)
        dl_b = __fdividef(lyr.z0 - p.z, p.uz);

      if (p.uz != 0.0f && p.s > dl_b) {
        p.sleft = (p.s - dl_b) * lyr.mut;
        p.s = dl_b;
        p.x += p.s * p.ux;
        p.y += p.s * p.uy;
        p.z += p.s * p.uz;
        gpu_cross_boundary_f32(&p, layer, num_layers, inv_dr, inv_da, nr, na,
                               &rng, target_Rd_ra, target_Tt_ra);
      } else {
        p.x += p.s * p.ux;
        p.y += p.s * p.uy;
        p.z += p.s * p.uz;

        int iz = (int)(p.z * inv_dz);
        if (iz > nz - 1)
          iz = nz - 1;
        else if (iz < 0)
          iz = 0;

        int ir = (int)(sqrtf(p.x * p.x + p.y * p.y) * inv_dr);
        if (ir > nr - 1)
          ir = nr - 1;

        float dwa = p.w * lyr.mua_over_mut;
        p.w -= dwa;

        if (SHARED_MODE == 2 && ir < hot_nr && iz < hot_nz) {
          atomicAdd(&s_mem[ir * hot_nz + iz], (double)dwa);
        } else {
          atomicAdd(&target_A_rz[(size_t)ir * (size_t)nz + (size_t)iz],
                    (double)dwa);
        }

        gpu_spin_f32(&lyr, &p, &rng);
      }
    }

    if (p.w < wth && !p.dead) {
      if (p.w <= 0.0f)
        p.dead = 1;
      else if (gpu_random_f32(&rng) < (float)CHANCE)
        p.w *= (float)(1.0 / CHANCE);
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
        atomicAdd(&d_A_rz[(size_t)ir * (size_t)nz + (size_t)iz], val);
      }
    }
  }
}

/* =========================================================================
 *  FULL DOUBLE-PRECISION (FP64) KERNEL (--fp64)
 * ========================================================================= */

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

__device__ __forceinline__ void gpu_spin(double g, PhotonStruct *p,
                                         uint64_t *rng)
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
                   double da, int nr, int na, uint64_t *rng,
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
        int ir = (ird > nr - 1) ? (nr - 1) : (int)ird;
        double iad = acos(-p->uz) / da;
        int ia = (iad > na - 1) ? (na - 1) : (int)iad;
        atomicAdd(&target_Rd_ra[(size_t)ir * (size_t)na + (size_t)ia], p->w);
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
        int ir = (ird > nr - 1) ? (nr - 1) : (int)ird;
        double iad = acos(p->uz) / da;
        int ia = (iad > na - 1) ? (na - 1) : (int)iad;
        atomicAdd(&target_Tt_ra[(size_t)ir * (size_t)na + (size_t)ia], p->w);
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

template <int SHARED_MODE>
__global__ void mcml_photon_kernel_fp64(
    unsigned long long num_photons, double rsp, double wth, double dz,
    double dr, double da, int nz, int nr, int na, short num_layers,
    uint64_t base_seed, unsigned long long *__restrict__ d_photon_counter,
    double *__restrict__ d_A_rz, double *__restrict__ d_Rd_ra,
    double *__restrict__ d_Tt_ra)
{
  extern __shared__ double s_mem[];
  int arz_size = nr * nz;
  int rdra_size = nr * na;
  int hot_nr = (nr < HOTSPOT_DIM) ? nr : HOTSPOT_DIM;
  int hot_nz = (nz < HOTSPOT_DIM) ? nz : HOTSPOT_DIM;

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
        int iz = (izd > nz - 1) ? (nz - 1) : (int)izd;
        double ird = sqrt(p.x * p.x + p.y * p.y) / dr;
        int ir = (ird > nr - 1) ? (nr - 1) : (int)ird;

        double dwa = p.w * mua / mut;
        p.w -= dwa;

        if (SHARED_MODE == 2 && ir < hot_nr && iz < hot_nz) {
          atomicAdd(&s_mem[ir * hot_nz + iz], dwa);
        } else {
          atomicAdd(&target_A_rz[(size_t)ir * (size_t)nz + (size_t)iz], dwa);
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
        atomicAdd(&d_A_rz[(size_t)ir * (size_t)nz + (size_t)iz], val);
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
  printf("[Backend] NVIDIA CUDA GPU (%s, Compute Capability %d.%d, %d SMs, %s)\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount,
         g_use_fp64 ? "Full FP64" : "Mixed FP32/FP64");

  size_t layers_bytes = (size_t)(In_Ptr->num_layers + 2) * sizeof(LayerStruct);
  if (cudaMemcpyToSymbol(d_layers, In_Ptr->layerspecs, layers_bytes) !=
      cudaSuccess) {
    return 0;
  }

  GpuLayerF32 h_layers_f32[MAX_GPU_LAYERS];
  for (int i = 0; i < In_Ptr->num_layers + 2; i++) {
    LayerStruct *src = &In_Ptr->layerspecs[i];
    double mut = src->mua + src->mus;
    double g = src->g;
    h_layers_f32[i].z0 = (float)src->z0;
    h_layers_f32[i].z1 = (float)src->z1;
    h_layers_f32[i].n = (float)src->n;
    h_layers_f32[i].mua = (float)src->mua;
    h_layers_f32[i].mus = (float)src->mus;
    h_layers_f32[i].mut = (float)mut;
    h_layers_f32[i].inv_mut = (mut > 0.0) ? (float)(1.0 / mut) : 0.0f;
    h_layers_f32[i].mua_over_mut =
        (mut > 0.0) ? (float)(src->mua / mut) : 0.0f;
    h_layers_f32[i].g = (float)g;
    h_layers_f32[i].one_minus_g2 = (float)(1.0 - g * g);
    h_layers_f32[i].one_plus_g2 = (float)(1.0 + g * g);
    h_layers_f32[i].one_minus_g = (float)(1.0 - g);
    h_layers_f32[i].two_g = (float)(2.0 * g);
    h_layers_f32[i].inv_2g = (g != 0.0) ? (float)(1.0 / (2.0 * g)) : 0.0f;
    h_layers_f32[i].cos_crit0 = (float)src->cos_crit0;
    h_layers_f32[i].cos_crit1 = (float)src->cos_crit1;
  }
  if (cudaMemcpyToSymbol(d_layers_f32, h_layers_f32,
                         (size_t)(In_Ptr->num_layers + 2) *
                             sizeof(GpuLayerF32)) != cudaSuccess) {
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
  size_t shared_bytes = 0;
  int shared_mode = 2;
  if (total_grid_elems <= MAX_SHARED_DOUBLES) {
    shared_mode = 1;
    shared_bytes = total_grid_elems * sizeof(double);
  } else {
    int hot_nr = (In_Ptr->nr < HOTSPOT_DIM) ? In_Ptr->nr : HOTSPOT_DIM;
    int hot_nz = (In_Ptr->nz < HOTSPOT_DIM) ? In_Ptr->nz : HOTSPOT_DIM;
    shared_bytes = (size_t)(hot_nr * hot_nz) * sizeof(double);
  }

  if (g_use_fp64) {
    if (shared_mode == 1) {
      mcml_photon_kernel_fp64<1><<<blocks, threads_per_block, shared_bytes>>>(
          (unsigned long long)In_Ptr->num_photons, Out_Ptr->Rsp, In_Ptr->Wth,
          In_Ptr->dz, In_Ptr->dr, In_Ptr->da, In_Ptr->nz, In_Ptr->nr,
          In_Ptr->na, In_Ptr->num_layers, seed, d_photon_counter, d_A_rz,
          d_Rd_ra, d_Tt_ra);
    } else {
      mcml_photon_kernel_fp64<2><<<blocks, threads_per_block, shared_bytes>>>(
          (unsigned long long)In_Ptr->num_photons, Out_Ptr->Rsp, In_Ptr->Wth,
          In_Ptr->dz, In_Ptr->dr, In_Ptr->da, In_Ptr->nz, In_Ptr->nr,
          In_Ptr->na, In_Ptr->num_layers, seed, d_photon_counter, d_A_rz,
          d_Rd_ra, d_Tt_ra);
    }
  } else {
    float inv_dz = (float)(1.0 / In_Ptr->dz);
    float inv_dr = (float)(1.0 / In_Ptr->dr);
    float inv_da = (float)(1.0 / In_Ptr->da);
    if (shared_mode == 1) {
      mcml_photon_kernel_fp32<1><<<blocks, threads_per_block, shared_bytes>>>(
          (unsigned long long)In_Ptr->num_photons, (float)Out_Ptr->Rsp,
          (float)In_Ptr->Wth, inv_dz, inv_dr, inv_da, In_Ptr->nz, In_Ptr->nr,
          In_Ptr->na, In_Ptr->num_layers, seed, d_photon_counter, d_A_rz,
          d_Rd_ra, d_Tt_ra);
    } else {
      mcml_photon_kernel_fp32<2><<<blocks, threads_per_block, shared_bytes>>>(
          (unsigned long long)In_Ptr->num_photons, (float)Out_Ptr->Rsp,
          (float)In_Ptr->Wth, inv_dz, inv_dr, inv_da, In_Ptr->nz, In_Ptr->nr,
          In_Ptr->na, In_Ptr->num_layers, seed, d_photon_counter, d_A_rz,
          d_Rd_ra, d_Tt_ra);
    }
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

  for (int ir = 0; ir < In_Ptr->nr; ir++) {
    for (int iz = 0; iz < In_Ptr->nz; iz++) {
      Out_Ptr->A_rz[ir][iz] +=
          h_A_rz[(size_t)ir * (size_t)In_Ptr->nz + (size_t)iz];
    }
    for (int ia = 0; ia < In_Ptr->na; ia++) {
      Out_Ptr->Rd_ra[ir][ia] +=
          h_Rd_ra[(size_t)ir * (size_t)In_Ptr->na + (size_t)ia];
      Out_Ptr->Tt_ra[ir][ia] +=
          h_Tt_ra[(size_t)ir * (size_t)In_Ptr->na + (size_t)ia];
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

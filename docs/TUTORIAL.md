# Local Build, Execution & Usage Tutorial

This guide provides a complete walkthrough of what `mcml-gpu` does, how to compile it with **OpenMP** and **NVIDIA CUDA**, how to select CPU thread counts or offload to a GPU, and how to analyze simulation outputs.

---

## 1. What Does This Project Do?

`mcml-gpu` is a high-performance **Monte Carlo solver for the steady-state Radiative Transport Equation (RTE) in multi-layered biological and tumoral tissues**, combining the classic MCML multi-layer formulation (Wang, Jacques, & Zheng, 1995) with the GPU parallelization and **Photodynamic Therapy (PDT)** tumoral dosimetry framework developed by **N. López-Marín, F. Corrales-Machín, E. Castro-Alvarez, and R. Rodríguez-Fernández** (*Revista Cubana de Física*, 33(2), 122–126, 2016) and **F. Corrales Machín** (*B.Sc. Thesis, University of Havana*, 2015).

In Photodynamic Therapy (PDT) for skin cancer (such as Basal Cell Carcinoma irradiated at $\lambda = 630\text{ nm}$), accurate spatial dosimetry of the absorbed light dose $A(r, z)$ is critical. Because biological tissues exhibit strong forward-peaked scattering ($g \approx 0.8\text{–}0.95$, $\mu_s \gg \mu_a$), neither the Beer-Lambert law nor the Diffusion Approximation is valid near the tissue surface or layer boundaries.

When a collimated laser beam strikes a layered medium, photon packets in `mcml-gpu` undergo:
1. **Specular Reflection ($R_{sp}$)** at the top ambient-medium interface due to refractive index mismatch.
2. **Stochastic Step Sampling (Hop)**: Photons travel a random free-path distance $s = -\ln(\xi)/(\mu_a + \mu_s)$ between interaction sites.
3. **Energy Deposition (Drop)**: At each interaction site, a fraction of the photon packet's weight $\Delta w = w \cdot \mu_a / (\mu_a + \mu_s)$ is absorbed and accumulated into a cylindrical ring grid $A(r, z)$ normalized by ring volume $\Delta V(r) = \pi \Delta r (2r + \Delta r)\Delta z$.
4. **Anisotropic Scattering (Spin)**: A new deflection angle $\Theta$ is sampled from the Henyey-Greenstein phase function governed by anisotropy $g = \langle\cos\Theta\rangle \in [0, 1]$.
5. **Fresnel Boundary Crossing**: At internal layer interfaces, Snell's law and Fresnel equations govern reflection, refraction, and total internal reflection (TIR).
6. **Russian Roulette Termination**: Low-weight photon packets ($w < 10^{-4}$) undergo unbiased Russian roulette (10% survival probability with $10\times$ weight boost) to strictly conserve total radiative energy ($R_{sp} + R_d + A + T_t = 1.0$).

---

## 2. How to Compile Locally (OpenMP + CUDA)

### Prerequisites
- **OS**: Linux, macOS, or Windows (WSL2)
- **Host Compiler**: `gcc` ($\ge 9.0$) or `clang` ($\ge 10.0$) with OpenMP support
- **GPU Compiler (Optional, Auto-Detected)**: NVIDIA CUDA Toolkit (`nvcc`, targets `sm_75` by default; configurable via `-DCMAKE_CUDA_ARCHITECTURES=<arch>`)
- **Build System**: `cmake` ($\ge 3.18$) and `make`

### Step-by-Step Build

```bash
# 1. Clone and enter the repository
git clone https://github.com/frankcm/mcml-gpu.git
cd mcml-gpu

# 2. Configure the out-of-source Release build
#    CMake automatically detects OpenMP and nvcc
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 3. Compile all targets in parallel
cmake --build build --parallel
```

> [!TIP]
> To build **CPU-only** on a machine with CUDA installed, pass `-DENABLE_CUDA=OFF`:
> ```bash
> cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
> ```

---

## 3. Selecting CPU Core Count (OpenMP) vs. GPU (CUDA)

The `mcml` executable supports command-line flags and environment variables to switch backends and thread counts cleanly:

### Option A: Run on CPU with OpenMP

You can set the exact number of CPU threads in **two ways**:

1. **Using the `--threads <N>` (or `-t <N>`) CLI flag**:
   ```bash
   # Run with 12 threads
   ./build/mcml --cpu --threads 12 data/test.mci

   # Run with 6 threads
   ./build/mcml --cpu --threads 6 data/test.mci

   # Run single-threaded (1 core)
   ./build/mcml --cpu --threads 1 data/test.mci
   ```
2. **Using the standard `OMP_NUM_THREADS` environment variable**:
   ```bash
   OMP_NUM_THREADS=8 ./build/mcml --cpu data/test.mci
   ```
   *(If you omit `--threads` and `OMP_NUM_THREADS`, OpenMP automatically uses all available logical cores on the host machine).*

### Option B: Run on NVIDIA GPU (CUDA)

Pass `--gpu` (or `--cuda`, or set `MCML_BACKEND=cuda`):

```bash
# Run quick smoke test on the NVIDIA CUDA GPU
./build/mcml --gpu data/test.mci

# Run the Basal Cell Carcinoma (630 nm, 10^9 photons) simulation on the GPU
./build/mcml --gpu data/bcc_tumor_630nm.mci
```

When running on the GPU, `mcml` prints the detected hardware profile:
```text
[Backend] NVIDIA CUDA GPU (NVIDIA GeForce RTX 2060, Compute Capability 7.5, 30 SMs)
```

---

## 4. Running Automated Tests & Benchmarks

### Run Unit & Integration Tests (CTest)
```bash
ctest --test-dir build --output-on-failure --verbose
```
This runs:
1. `unit_physics_tests`: Verifies PRNG uniformity, analytical Fresnel reflectance, total internal reflection, Henyey-Greenstein moments, and energy conservation.
2. `integration_smoke_test_cpu`: Verifies the OpenMP CPU backend.
3. `integration_smoke_test_gpu`: Verifies the NVIDIA CUDA GPU backend (when built with CUDA).

### Run the Scaling & GPU Benchmark Suite
```bash
# Run default full-scale benchmark (10^9 photons on BCC 630 nm model)
./benchmarks/run_benchmarks.sh

# Or run a fast benchmark with a custom photon count (e.g., 10^6 photons)
MCML_BENCH_PHOTONS=1000000 MCML_WEAK_PHOTONS_PER_THREAD=250000 ./benchmarks/run_benchmarks.sh
```

---

## 5. Running Custom Simulations Without Committing Results

1. Create a local `results/` directory (automatically ignored by [`.gitignore`](../.gitignore)):
   ```bash
   mkdir -p results
   cp data/bcc_tumor_630nm.mci results/custom_run.mci
   ```
2. Edit `results/custom_run.mci` so the output file path points inside `results/`:
   ```text
   1.0
   1
   results/custom_run.mco      A
   1000000000
   0.0005  0.0005
   400     800     1
   1
   1.0
   1.37    1.7     125.0   0.80    0.20
   1.0
   ```
3. Run the simulation and verify radiative energy conservation:
   ```bash
   ./build/mcml --gpu results/custom_run.mci
   python3 scripts/validate_mco.py results/custom_run.mco
   ```

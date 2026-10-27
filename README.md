# mcml-gpu

[![CI](https://github.com/machinfc/mcml-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/machinfc/mcml-gpu/actions/workflows/ci.yml)
[![Standard: C99](https://img.shields.io/badge/C-C99-00599C.svg?logo=c&logoColor=white)](https://en.wikipedia.org/wiki/C99)
[![CUDA: sm_75+](https://img.shields.io/badge/CUDA-sm__75%2B-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Parallel: OpenMP](https://img.shields.io/badge/Parallel-OpenMP-success.svg)](https://www.openmp.org/)
[![Build: CMake](https://img.shields.io/badge/Build-CMake%20%E2%89%A5%203.18-064F8C.svg?logo=cmake)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 1. What It Is & Purpose

**High-performance Monte Carlo solver for the steady-state Radiative Transport Equation (RTE) in multi-layered biological and tumoral tissues, accelerated with OpenMP and NVIDIA CUDA.**  
Designed for **Photodynamic Therapy (PDT) light dosimetry** and tissue optics where analytical Beer-Lambert and Diffusion approximations fail, extending the GPU radiative transfer framework of [López-Marín, Corrales-Machín et al. (2016)](#references--citation) and classic [MCML (Wang et al., 1995)](#references--citation).

---

## 2. Build & Run

### Prerequisites

| Dependency | Minimum Version | Purpose |
| :--- | :--- | :--- |
| **GCC** or **Clang** | GCC ≥ 9.0 / Clang ≥ 10.0 | C99/C++ host compiler with OpenMP runtime (`libgomp` / `libomp`) |
| **CMake** | ≥ 3.18 | Build system with automatic CUDA & OpenMP detection |
| **NVIDIA CUDA Toolkit** | ≥ 11.0 *(optional)* | Compiles `src/mcml/mcmlgpu.cu` (auto-detected; falls back to CPU-only if absent) |
| **Python 3** | ≥ 3.8 *(optional)* | Automated `.mco` radiative energy conservation verifier |

### Compile & Execute (Copy-Pasteable)

```bash
# 1. Clone and enter the repository
git clone https://github.com/machinfc/mcml-gpu.git
cd mcml-gpu

# 2. Configure and build in Release mode (auto-detects OpenMP and NVIDIA nvcc)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 3. Quick smoke test on CPU (OpenMP) or NVIDIA GPU (CUDA)
./build/mcml --cpu --threads 12 data/test.mci
./build/mcml --gpu data/test.mci

# 4. Run Basal Cell Carcinoma (BCC) PDT dosimetry model at 630 nm (10^9 photons)
./build/mcml --gpu data/bcc_tumor_630nm.mci

# 5. Convolve pencil-beam .mco impulse response with a finite Gaussian or flat-top laser beam
./build/conv build/test.mco build/test_convolved.dat --beam gaussian --power 1.0 --radius 0.1
```
---

## 3. Verification, Testing & Benchmarks

### Running the Automated Test Suite

The CTest suite runs unit tests on the optical physics engine (`tests/test_physics.c` — PRNG uniformity, analytical Fresnel specular reflectance, total internal reflection, Henyey-Greenstein first moment $\langle\cos\Theta\rangle = g$, 32-bit grid allocation, modified Bessel $e^{-x}I_0(x)$ scaling, and flat/Gaussian finite-beam radial convolution), an OpenMP CPU integration test, a CUDA GPU integration test, and a finite-beam `conv` CLI integration test:

```bash
# Run all unit, OpenMP CPU, CUDA GPU, and finite-beam convolution tests
ctest --test-dir build --output-on-failure --verbose

# Validate radiative energy conservation (Rsp + Rd + A + Tt = 1.0) on any output file
python3 scripts/validate_mco.py build/test.mco
```

### Physical Accuracy Validation — Basal Cell Carcinoma at $\lambda = 630\text{ nm}$

Both the OpenMP CPU backend and the NVIDIA CUDA GPU kernel strictly conserve radiative energy ($R_{sp} + R_d + A + T_t = 1.000000$) and agree within Monte Carlo statistical variability ($O(N^{-1/2})$):

| Physical Observable | OpenMP CPU | NVIDIA CUDA GPU | Absolute Difference | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Specular Reflectance ($R_{sp}$)** | `0.024373` | `0.024373` | `0.000000` | ✅ Exact |
| **Diffuse Reflectance ($R_d$)** | `0.321482` | `0.321516` | `0.000034` | ✅ Within $1\sigma$ |
| **Total Absorption ($A$)** | `0.575114` | `0.575095` | `0.000019` | ✅ Within $1\sigma$ |
| **Total Transmittance ($T_t$)** | `0.079031` | `0.079016` | `0.000015` | ✅ Within $1\sigma$ |
| **Total Energy ($R_{sp}+R_d+A+T_t$)** | `1.000000` | `1.000000` | `0.000000` | ✅ Conserved |

---

### Engineering Benchmarks — Basal Cell Carcinoma ($\lambda = 630\text{ nm}$) at $N = 10^9$ Photons

All benchmarks below evaluate the **Basal Cell Carcinoma (BCC) Photodynamic Therapy model at $\lambda = 630\text{ nm}$** specified in [Corrales-Machín (2015)](#references--citation) and [López-Marín et al. (2016)](#references--citation) (`data/bcc_tumor_630nm.mci`):
- **Optical Properties**: $n = 1.37$, $\mu_a = 1.7\text{ cm}^{-1}$, $\mu_s = 125.0\text{ cm}^{-1}$, $g = 0.80$, thickness $d = 0.20\text{ cm}$, radius $r_{\max} = 0.40\text{ cm}$
- **Cylindrical Scoring Grid**: $\Delta z = 0.0005\text{ cm}$, $\Delta r = 0.0005\text{ cm}$ ($n_z = 400 \times n_r = 800$ bins)
- **Workload**: **$N = 10^9$ (`1,000,000,000`) photon packets** (configured by default in [`benchmarks/run_benchmarks.sh`](benchmarks/run_benchmarks.sh) and [`data/bcc_tumor_630nm.mci`](data/bcc_tumor_630nm.mci)).

#### A. Before vs. After Optimization ($N = 10^9$ Photons — Contemporary Hardware)
- **Host CPU**: Intel® Core™ i7-9750H @ 2.60 GHz (6 physical cores / 12 logical threads)
- **Device GPU**: NVIDIA GeForce RTX™ 2060 (6 GB GDDR6, Turing `sm_75`, 30 SMs / 1,920 CUDA cores, CUDA 13.1)

| Execution Backend | Compiler / Hardware Configuration | Wall Time ($10^9$ Photons) | Throughput $C$ (photons/s) | Speedup vs. 1T |
| :--- | :--- | :---: | :---: | :---: |
| **1. Serial CPU (`S02` Baseline)** | `GCC -O3 -march=native -ffast-math` (1 Thread) | `6,367.0 s` (~1.77 hr) | `1.57 × 10⁵ ph/s` | `1.00×` |
| **2. OpenMP Physical Cores** | `OpenMP` (`--cpu --threads 6`, 6 Physical Cores) | `1,747.0 s` (~29.1 min) | `5.72 × 10⁵ ph/s` | `3.64×` |
| **3. OpenMP Full SMT** | `OpenMP` (`--cpu --threads 12`, 12 Logical Threads) | `1,632.0 s` (~27.2 min) | `6.13 × 10⁵ ph/s` | `3.90×` |
| **4. NVIDIA CUDA (Initial FP64)** | `GeForce RTX 2060` (`sm_75`, Static Stride, Global Atomics) | `567.3 s` (~9.5 min) | `1.76 × 10⁶ ph/s` | `11.22×` |
| **5. NVIDIA CUDA (Phase 1 FP64)** | `GeForce RTX 2060` (`--gpu --fp64`, Persistent Queue + `64×64` Shared Hot-Spot) | `498.9 s` (~8.3 min) | `2.00 × 10⁶ ph/s` | `12.76×` |
| **6. NVIDIA CUDA (Phase 2 Mixed)** | `GeForce RTX 2060` (`--gpu --fp32`, SFU Intrinsics + `FP64` Accumulators) | **`75.1 s` (~1.25 min)** | **`1.33 × 10⁷ ph/s`** | **`84.78×` (`330.1×` vs `S02`)** |

#### B. OpenMP Strong Scaling ($N = 10^9$ Photons — Intel® Core™ i7-9750H)

| Threads ($P$) | Photons per Thread | Wall Time (s) | Throughput $C$ (photons/s) | Speedup vs. 1T | Parallel Efficiency |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `1,000,000,000` | `6,367.0 s` | `1.57 × 10⁵ ph/s` | `1.00×` | `100.0%` |
| **2** | `500,000,000` | `3,556.0 s` | `2.81 × 10⁵ ph/s` | `1.79×` | `89.5%` |
| **4** | `250,000,000` | `2,172.0 s` | `4.60 × 10⁵ ph/s` | `2.93×` | `73.3%` |
| **6 (Physical)** | `166,666,667` | `1,747.0 s` | `5.72 × 10⁵ ph/s` | `3.64×` | `60.7%` |
| **8 (SMT)** | `125,000,000` | `1,680.0 s` | `5.95 × 10⁵ ph/s` | `3.79×` | `47.4%` |
| **12 (Full SMT)** | `83,333,333` | `1,632.0 s` | `6.13 × 10⁵ ph/s` | `3.90×` | `32.5%` |

#### C. OpenMP Weak Scaling ($10^8$ Photons / Thread — Intel® Core™ i7-9750H)

| Threads ($P$) | Total Photons ($N$) | Wall Time (s) | Throughput $C$ (photons/s) | Weak Scaling Efficiency |
| :---: | :---: | :---: | :---: | :---: |
| **1** | `1.0 × 10⁸` | `636.7 s` | `1.57 × 10⁵ ph/s` | `100.0%` |
| **2** | `2.0 × 10⁸` | `711.2 s` | `2.81 × 10⁵ ph/s` | `89.5%` |
| **4** | `4.0 × 10⁸` | `868.8 s` | `4.60 × 10⁵ ph/s` | `73.3%` |
| **6 (Physical)** | `6.0 × 10⁸` | `1,048.2 s` | `5.72 × 10⁵ ph/s` | `60.7%` |
| **8 (SMT)** | `8.0 × 10⁸` | `1,344.0 s` | `5.95 × 10⁵ ph/s` | `47.4%` |
| **12 (Full SMT)** | `1.2 × 10⁹` | `1,958.4 s` | `6.13 × 10⁵ ph/s` | `32.5%` |

#### D. Historical Reference Benchmark ($N = 10^9$ Photons — Thesis & Published Paper)
Measured in [Corrales-Machín (2015)](#references--citation) and [López-Marín et al. (2016)](#references--citation) across $N \in [10^6, 10^{10}]$ photon packets on the original HPC server:
- **Host CPU**: 2× AMD Opteron™ 6800 Series (24 cores total, 128 GB RAM)
- **Device GPUs**: 2× NVIDIA Tesla™ C2050 / C2025 (Fermi architecture, $2 \times 448$ CUDA cores, 4 GB GDDR5 each)
- **Power-Law Fit**: $t(N) = \frac{N^m}{C}$, where $m \approx 1.000$ confirms strict linear complexity and $C$ ($\text{s}^{-1}$) is the sustained throughput:

| Implementation Variant | Architecture & Algorithmic Features | Slope ($m$) | Throughput $C$ ($\text{photons/s}$) | Wall Time ($N = 10^9$) | Speedup vs. `S02` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`S02` (Serial CPU)** | Single-core ANSI C baseline on AMD Opteron | `0.972 ± 0.002` | `(2.12 ± 0.08) × 10⁴` | `24,792 s` (~6.89 hr) | `1.0×` |
| **`A01` (CUDA Host-Loop)** | 1 photon/thread per launch; outer loop on CPU | `1.00005 ± 0.00001` | `(1.92088 ± 0.00008) × 10⁶` | `521.1 s` (~8.7 min) | `47.6×` |
| **`A02` (CUDA In-Kernel)** | Persistent photon loop inside GPU kernel (`MP²` blocks × 512 threads) | `0.9996 ± 0.0001` | `(3.095 ± 0.007) × 10⁶` | `320.4 s` (~5.3 min) | `77.4×` |
| **`A06` (CUDA Fast-Math)** | `A02` + single-precision hardware ALU intrinsics (`--use_fast_math`) | `0.9996 ± 0.0001` | `(4.89 ± 0.01) × 10⁶` | `202.8 s` (~3.4 min) | `122.2×` |
| **`A09` / `A07` (Dual GPU)** | Dual Tesla C2050 + Fast-Math ALU + Pinned Host Memory | `0.9992 ± 0.0001` | **`(11.51 ± 0.04) × 10⁶`** | **`85.4 s` (~1.4 min)** | **`290.3×`** |

*(Note: `A06` and `A09` traced single-precision monolayer trajectories without refractive boundary crossings, whereas `mcml-gpu` v1.4 traces multi-layer trajectories with exact Fresnel/Snell boundary optics, angularly resolved reflectance/transmittance scoring, and 64-bit `FP64` energy accumulators).*

---

## 4. Project Status & Honest Technical Limitations

### ✅ Implemented & Verified
- **Multi-Layer Generalization of the Tumoral GPU Transport Model**: Implements the recommendations from [Corrales-Machín (2015)](#references--citation) and [López-Marín et al. (2016)](#references--citation), extending single-layer Basal Cell Carcinoma ($\lambda = 630\text{ nm}$) transport to arbitrary multi-layered tissues with Fresnel/Snell boundary crossings and critical-angle total internal reflection (TIR).
- **Persistent In-Loop Photon Respawning & Dynamic Batch Work-Queue (`src/mcml/mcmlgpu.cu`)**: Resolves SIMT warp lifetime divergence (where 31 lanes in a 32-thread warp previously idled waiting for the longest-lived photon packet to finish its `while (!p.dead)` loop). Terminated lanes immediately claim and initialize a fresh photon from an atomic batch queue (`WORK_BATCH_SIZE = 64`), keeping all 32 warp lanes continuously active.
- **Shared-Memory Hot-Spot Sub-Grid Privatization (`64 × 64` Sub-Grid Cache)**: Resolves shared-memory spillover on large clinical grids (`nr * nz > 4,096`, such as the $400 \times 800$ BCC grid in `data/bcc_tumor_630nm.mci`). Each thread block caches the high-contention subsurface pencil-beam entry core ($i_r < 64, i_z < 64$, 32 KB `__shared__` memory) in on-chip SRAM and flushes once to global memory on kernel exit.
- **Mixed-Precision Fast-Math GPU Kinematics (`--fp32` Default) & Full `FP64` Mode (`--fp64`)**: Unlocks the 32:1 `FP32`-to-`FP64` throughput advantage and Special Function Units (`__logf`, `__sincosf`, `__fdividef`, `rsqrtf`) on consumer NVIDIA Turing GPUs while accumulating all absorbed, reflected, and transmitted energy weights in strict 64-bit `FP64` (`double`). Reduces $10^9$-photon clinical simulation time on a single GeForce RTX 2060 from `498.9 s` (`--fp64`) to **`75.1 s`** (`--fp32`, **`84.8×`** faster than single-core i7-9750H and **`330.1×`** faster than `S02`).
- **32-Bit Cylindrical Grid Indexing (`int` / `size_t`)**: Eliminates the legacy 16-bit `short` grid ceiling (`32,767` bins) across `InputStruct`, memory allocators (`mcmlnr.c`), I/O parsers (`mcmlio.c`), and CPU/GPU kernels.
- **Finite-Beam Convolution Post-Processor (`src/conv/` $\rightarrow$ `build/conv`)**: Standalone C99 utility and library routines that convolve pencil-beam `.mco` impulse responses (`A_rz`, `Rd_r`, `Tt_r`) against finite-radius Gaussian ($\text{TEM}_{00}$) and circular flat-top laser beams using adaptive Simpson quadrature (`qtrap`/`qsimp`) and scaled modified Bessel functions ($e^{-x}I_0(x)$). Replaces all deprecated C89 `gets()` calls with bounds-checked `fgets()` and fixes two legacy bugs in `convconv.c` (`A_rzInterp` stub return and missing `Gaussian` branch in `ConvA_rz`).
- **Automated CTest & CI Pipeline**: Unit and GPU verification tests for Fresnel optics, Henyey-Greenstein moments, 32-bit grid allocation ($> 32,767$ bins), modified Bessel and radial beam convolution integrals, large-grid hot-spot privatization, and radiative energy conservation across both `Mixed FP32/FP64` and `Full FP64` modes.
- **Strict Local Data Isolation**: `.gitignore` blocks all personal simulation outputs (`*.mco`, `*.dat`, `results/`, `benchmarks/results/`).

### ❌ Not Yet Optimized / Known Architectural Bottlenecks
- **Inter-Layer SIMT Branch Divergence within Active Steps**: While persistent in-loop respawning eliminates idle-lane lifetime divergence, threads within a warp that simultaneously reside in different tissue layers or take different boundary vs. interior scattering branches still serialize conditional execution paths within a single step.
- **Outer-Radius Global Atomic Fallback ($i_r \ge 64$ or $i_z \ge 64$)**: While the $64 \times 64$ `__shared__` hot-spot cache absorbs the highest-contention pencil-beam entry core, diffuse photons scattering beyond $r \ge 64\Delta r$ or $z \ge 64\Delta z$ still issue global `atomicAdd(double*, double)` transactions to L2 cache.
- **Single-GPU Execution (No Multi-GPU / MPI Domain Decomposition)**: Currently dispatches to a single CUDA device; multi-GPU node scaling requires host-side OpenMP/MPI domain partitioning across devices.
- **Planar Multi-Layered Geometry Only ($z$-Axis Interfaces)**: Tissue interfaces are modeled as infinite horizontal planes ($z = z_k$); curved 3D tumor boundaries require tetrahedral mesh or voxelized ray-tracing.
- **Time-Independent (Continuous-Wave) RTE Solver**: Solves the stationary RTE ($\partial I/\partial t = 0$); ultrashort pulsed laser dosimetry requires time-of-flight (`dt`) histogram binning.

---

## Repository Structure

```text
mcml-gpu/
├── .github/workflows/ci.yml     # Multi-compiler CI (GCC/Clang, Release/Debug, CTest)
├── benchmarks/
│   ├── README.md                # Benchmark methodology & scaling definitions
│   └── run_benchmarks.sh        # Automated OpenMP & CUDA GPU benchmark runner (10^9 default)
├── data/
│   ├── README.md                # Dataset overview & optical parameter table
│   ├── test.mci                 # Fast smoke-test configuration (100K photons)
│   ├── validation.mci           # Wang et al. (1995) reference benchmark (1M photons)
│   ├── bcc_tumor_630nm.mci      # Basal Cell Carcinoma PDT model at 630 nm (10^9 photons)
│   └── example.mci              # High-resolution 3-layer production run (100M photons, 800x800 grid)
├── docs/
│   ├── TUTORIAL.md              # Step-by-step local build, CPU/GPU execution & usage guide
│   ├── ARCHITECTURE.md          # RTE physics, Beer-Lambert/Diffusion comparison & GPU architecture
│   └── INPUT_FORMAT.md          # Complete .mci input and .mco output specification
├── include/
│   ├── mcml.h                   # Core structs, constants, and public C/CUDA prototypes
│   ├── conv.h                   # Convolution post-processor declarations
│   └── extra.h                  # Numerical integration & modified Bessel helpers
├── scripts/
│   └── validate_mco.py          # Automated radiative energy conservation verifier
├── src/
│   ├── mcml/
│   │   ├── mcmlmain.c           # CLI entry point (--gpu, --cpu, --fp32, --fp64, --threads)
│   │   ├── mcmlgpu.cu           # NVIDIA CUDA kernel (Persistent queue, 64x64 shared hot-spot, Mixed FP32/FP64)
│   │   ├── mcmlgo.c             # OpenMP CPU Hop-Drop-Spin kernel & thread-safe ran3
│   │   ├── mcmlio.c             # Input parser, parameter validator & ASCII serializer
│   │   └── mcmlnr.c             # Zero-initialized vector/matrix memory allocators
│   └── conv/
│       ├── convmain.c           # Finite-beam convolution CLI tool (build/conv)
│       ├── convconv.c           # Gaussian & flat-top radial Hankel/Bessel convolution engine
│       └── extra.c              # Adaptive Simpson quadrature (qtrap/qsimp) & Bessel I0
├── tests/
│   └── test_physics.c           # C99 unit & physics verification test suite
├── .editorconfig
├── .gitignore                   # Excludes all local .mco/.dat simulation outputs & build files
├── CITATION.cff                 # GitHub citation metadata for published papers
├── CMakeLists.txt               # CMake >= 3.18 build (OpenMP + auto-detected CUDA sm_75)
├── CONTRIBUTING.md              # Contribution guidelines & engineering roadmap
└── LICENSE                      # MIT License + scientific attributions
```

---

## Documentation

- [**Local Build, CPU/GPU Execution & Usage Tutorial (`docs/TUTORIAL.md`)**](docs/TUTORIAL.md)
- [**Radiative Transfer Physics & GPU Architecture (`docs/ARCHITECTURE.md`)**](docs/ARCHITECTURE.md)
- [**Input (`.mci`) & Output (`.mco`) Specification (`docs/INPUT_FORMAT.md`)**](docs/INPUT_FORMAT.md)
- [**Benchmarking Guide (`benchmarks/README.md`)**](benchmarks/README.md)

---

## References & Citation

If you use `mcml-gpu` in academic or clinical dosimetry research, please cite:

1. **N. López-Marín, F. Corrales-Machín, E. Castro-Alvarez, and R. Rodríguez-Fernández**, "Light Transport in Biological Tissues Using GPU (*Transporte de luz en tejidos biológicos utilizando procesadores gráficos*)," *Revista Cubana de Física*, Vol. 33, No. 2, pp. 122–126 (2016).
2. **F. Corrales Machín**, "*Simulación del transporte de luz en una capa de tejido utilizando el método de Monte Carlo*," B.Sc. Thesis (*Trabajo de Diploma*), Facultad de Física, Universidad de La Habana (2015).
3. **L.-H. Wang, S. L. Jacques, and L.-Q. Zheng**, "MCML — Monte Carlo modeling of photon transport in multi-layered tissues," *Computer Methods and Programs in Biomedicine*, 47(2), 131–146 (1995).
4. **E. Salomatina, B. Jiang, J. Novak, and A. N. Yaroslavsky**, "Optical properties of normal and cancerous human skin in the visible and near-infrared spectral range," *Journal of Biomedical Optics*, 11(6), 064026 (2006).

```bibtex
@article{LopezMarin2016GPU,
  author  = {L{\'o}pez-Mar{\'i}n, N. and Corrales-Mach{\'i}n, F. and Castro-Alvarez, E. and Rodr{\'i}guez-Fern{\'a}ndez, R.},
  title   = {Light Transport in Biological Tissues Using {GPU}},
  journal = {Revista Cubana de F{\'i}sica},
  volume  = {33},
  number  = {2},
  pages   = {122--126},
  year    = {2016}
}
```

## License

Licensed under the [MIT License](LICENSE). Original MCML code Copyright © 1992 University of Texas M.D. Anderson Cancer Center.

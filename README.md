# mcml-gpu

[![CI](https://github.com/machinfc/mcml-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/machinfc/mcml-gpu/actions/workflows/ci.yml)
[![Standard: C99](https://img.shields.io/badge/C-C99-00599C.svg?logo=c&logoColor=white)](https://en.wikipedia.org/wiki/C99)
[![CUDA: sm_75+](https://img.shields.io/badge/CUDA-sm__75%2B-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Parallel: OpenMP](https://img.shields.io/badge/Parallel-OpenMP-success.svg)](https://www.openmp.org/)
[![Build: CMake](https://img.shields.io/badge/Build-CMake%20%E2%89%A5%203.18-064F8C.svg?logo=cmake)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 1. Overview & Purpose

**High-performance Monte Carlo solver for the steady-state Radiative Transport Equation (RTE) in multi-layered biological and tumoral tissues, accelerated with OpenMP and NVIDIA CUDA.**

Designed for **Photodynamic Therapy (PDT) light dosimetry** and biomedical tissue optics in regimes where analytical Beer-Lambert and Diffusion approximations break down (high anisotropy `g ≈ 0.80–0.95`, thin layers, and refractive boundaries). This project generalizes the GPU tumoral radiative transfer framework of [López-Marín, Corrales-Machín et al. (2016)](#references--citation) and classic [MCML (Wang et al., 1995)](#references--citation).

| Key Capability | Technical Implementation | Verified Result (`10⁹` Photons, BCC 630 nm) |
| :--- | :--- | :--- |
| **Multi-Layer RTE Physics** | Hop-Drop-Spin transport, Snell/Fresnel boundaries, TIR & Russian roulette | Energy conserved (`Rsp + Rd + A + Tt = 1.000000`) |
| **NVIDIA CUDA Engine** | Persistent warp work-queue, `64×64` `__shared__` cache, Mixed `FP32/FP64` SFU | **`75.1 s`** on RTX 2060 (**`84.8×`** vs 1T CPU, **`330.1×`** vs `S02`) |
| **OpenMP CPU Engine** | Thread-private `ran3` PRNG streams & private 2D scoring grids | **`1,632.0 s`** on 12T Core i7-9750H (**`3.90×`** vs 1T CPU) |
| **Finite-Beam Post-Processor** | Standalone CLI (`build/conv`) for Gaussian (`TEM₀₀`) & flat-top laser beams | Adaptive Simpson quadrature + scaled Bessel `exp(-x)I₀(x)` |

---

## 2. Build & Run

### Prerequisites

| Dependency | Minimum Version | Purpose |
| :--- | :--- | :--- |
| **GCC** or **Clang** | GCC ≥ 9.0 / Clang ≥ 10.0 | C99/C++ host compiler with OpenMP runtime (`libgomp` / `libomp`) |
| **CMake** | ≥ 3.18 | Build system with automatic CUDA & OpenMP detection |
| **NVIDIA CUDA Toolkit** | ≥ 11.0 *(optional)* | Compiles `src/mcml/mcmlgpu.cu` (falls back to CPU-only if absent) |
| **Python 3** | ≥ 3.8 *(optional)* | Automated `.mco` radiative energy conservation verifier |

### Quick Start (Copy-Pasteable)

```bash
# 1. Clone and enter the repository
git clone https://github.com/machinfc/mcml-gpu.git
cd mcml-gpu

# 2. Configure and compile in Release mode (auto-detects OpenMP and NVIDIA nvcc)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 3. Run quick smoke test on CPU (OpenMP) or NVIDIA GPU (CUDA)
./build/mcml --cpu --threads 12 data/test.mci
./build/mcml --gpu data/test.mci

# 4. Run Basal Cell Carcinoma (BCC) PDT dosimetry model at 630 nm (10^9 photons)
./build/mcml --gpu data/bcc_tumor_630nm.mci

# 5. Convolve pencil-beam .mco response with a finite Gaussian or flat-top beam
./build/conv build/test.mco build/test_conv.dat --beam gaussian --power 1.0 --radius 0.1
```

---

## 3. Verification, Testing & Benchmarks

### Automated Test Suite (CTest)

The test suite (`tests/test_physics.c`) verifies PRNG uniformity, analytical Fresnel specular reflectance, total internal reflection (TIR), Henyey-Greenstein first moment (`<cos Θ> = g`), 32-bit grid allocation (`nz = 40,000 > 32,767`), modified Bessel `exp(-x)I₀(x)` scaling, flat/Gaussian finite-beam radial convolution, and CPU/GPU radiative energy conservation:

```bash
# Run all unit, OpenMP CPU, CUDA GPU, and finite-beam convolution tests
ctest --test-dir build --output-on-failure --verbose

# Validate radiative energy conservation (Rsp + Rd + A + Tt = 1.0) on any .mco file
python3 scripts/validate_mco.py build/test.mco
```

### Physical Accuracy Validation — Basal Cell Carcinoma (`λ = 630 nm`)

Both the OpenMP CPU backend and the NVIDIA CUDA GPU kernel strictly conserve radiative energy (`Rsp + Rd + A + Tt = 1.000000`) and agree within Monte Carlo statistical uncertainty (`O(N⁻¹/²)`):

| Physical Observable | OpenMP CPU | NVIDIA CUDA GPU | Absolute Diff | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Specular Reflectance (`Rsp`)** | `0.024373` | `0.024373` | `0.000000` | ✅ Exact |
| **Diffuse Reflectance (`Rd`)** | `0.321482` | `0.321516` | `0.000034` | ✅ Within `1σ` |
| **Total Absorption (`A`)** | `0.575114` | `0.575095` | `0.000019` | ✅ Within `1σ` |
| **Total Transmittance (`Tt`)** | `0.079031` | `0.079016` | `0.000015` | ✅ Within `1σ` |
| **Total Energy (`Rsp + Rd + A + Tt`)** | `1.000000` | `1.000000` | `0.000000` | ✅ Conserved |

---

### Engineering Benchmarks — Basal Cell Carcinoma (`λ = 630 nm`, `N = 10⁹` Photons)

All benchmarks evaluate the **Basal Cell Carcinoma (BCC) Photodynamic Therapy model at `λ = 630 nm`** defined in [Corrales-Machín (2015)](#references--citation) and [López-Marín et al. (2016)](#references--citation) ([`data/bcc_tumor_630nm.mci`](data/bcc_tumor_630nm.mci)):
- **Optical Properties**: `n = 1.37`, `μₐ = 1.7 cm⁻¹`, `μₛ = 125.0 cm⁻¹`, `g = 0.80`, thickness `d = 0.20 cm`, radius `r_max = 0.40 cm`
- **Cylindrical Scoring Grid**: `Δz = 0.0005 cm`, `Δr = 0.0005 cm` (`nz = 400 × nr = 800` bins)
- **Workload**: **`N = 10⁹` (`1,000,000,000`) photon packets** (default in [`benchmarks/run_benchmarks.sh`](benchmarks/run_benchmarks.sh))

#### A. End-to-End Optimization Progression (`N = 10⁹` Photons — Contemporary Hardware)
- **Host CPU**: Intel® Core™ i7-9750H @ 2.60 GHz (6 physical cores / 12 logical threads)
- **Device GPU**: NVIDIA GeForce RTX™ 2060 (6 GB GDDR6, Turing `sm_75`, 30 SMs / 1,920 CUDA cores)

| Execution Stage | Hardware & Kernel Configuration | Wall Time (`10⁹`) | Throughput (`ph/s`) | Speedup vs. 1T |
| :--- | :--- | :---: | :---: | :---: |
| **1. Serial CPU (`S02`)** | Core i7-9750H (`1 Thread`, `-O3 -march=native`) | `6,367.0 s` (~1.77 hr) | `1.57 × 10⁵` | `1.00×` |
| **2. OpenMP Physical** | Core i7-9750H (`--cpu --threads 6`, 6 Cores) | `1,747.0 s` (~29.1 min) | `5.72 × 10⁵` | `3.64×` |
| **3. OpenMP Full SMT** | Core i7-9750H (`--cpu --threads 12`, 12 Threads) | `1,632.0 s` (~27.2 min) | `6.13 × 10⁵` | `3.90×` |
| **4. CUDA Baseline FP64** | RTX 2060 (`sm_75`, Static Stride, Global Atomics) | `567.3 s` (~9.5 min) | `1.76 × 10⁶` | `11.22×` |
| **5. CUDA Phase 1 FP64** | RTX 2060 (`--fp64`, Persistent Queue + `64×64` Shared) | `498.9 s` (~8.3 min) | `2.00 × 10⁶` | `12.76×` |
| **6. CUDA Phase 2 Mixed** | RTX 2060 (`--fp32`, SFU Intrinsics + `FP64` Scoring) | **`75.1 s` (~1.25 min)** | **`1.33 × 10⁷`** | **`84.78×`** |

*(Note: Stage 6 achieves an **`84.78×`** speedup over single-core Core i7-9750H and a **`330.1×`** speedup over the original `S02` serial baseline).*

#### B. OpenMP Strong Scaling (`N = 10⁹` Photons — Intel® Core™ i7-9750H)

| Threads (`P`) | Photons / Thread | Wall Time (s) | Throughput (`ph/s`) | Speedup vs. 1T | Parallel Efficiency |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `1,000,000,000` | `6,367.0 s` | `1.57 × 10⁵` | `1.00×` | `100.0%` |
| **2** | `500,000,000` | `3,556.0 s` | `2.81 × 10⁵` | `1.79×` | `89.5%` |
| **4** | `250,000,000` | `2,172.0 s` | `4.60 × 10⁵` | `2.93×` | `73.3%` |
| **6 (Physical)** | `166,666,667` | `1,747.0 s` | `5.72 × 10⁵` | `3.64×` | `60.7%` |
| **8 (SMT)** | `125,000,000` | `1,680.0 s` | `5.95 × 10⁵` | `3.79×` | `47.4%` |
| **12 (Full SMT)** | `83,333,333` | `1,632.0 s` | `6.13 × 10⁵` | `3.90×` | `32.5%` |

#### C. OpenMP Weak Scaling (`10⁸` Photons / Thread — Intel® Core™ i7-9750H)

| Threads (`P`) | Total Photons (`N`) | Wall Time (s) | Throughput (`ph/s`) | Weak Scaling Efficiency |
| :---: | :---: | :---: | :---: | :---: |
| **1** | `1.0 × 10⁸` | `636.7 s` | `1.57 × 10⁵` | `100.0%` |
| **2** | `2.0 × 10⁸` | `711.2 s` | `2.81 × 10⁵` | `89.5%` |
| **4** | `4.0 × 10⁸` | `868.8 s` | `4.60 × 10⁵` | `73.3%` |
| **6 (Physical)** | `6.0 × 10⁸` | `1,048.2 s` | `5.72 × 10⁵` | `60.7%` |
| **8 (SMT)** | `8.0 × 10⁸` | `1,344.0 s` | `5.95 × 10⁵` | `47.4%` |
| **12 (Full SMT)** | `1.2 × 10⁹` | `1,958.4 s` | `6.13 × 10⁵` | `32.5%` |

#### D. Historical Reference Benchmark (`N = 10⁹` Photons — Thesis & Published Paper)
Measured in [Corrales-Machín (2015)](#references--citation) and [López-Marín et al. (2016)](#references--citation) across `N ∈ [10⁶, 10¹⁰]` photon packets on the original HPC server:
- **Host CPU**: 2× AMD Opteron™ 6800 Series (24 cores total, 128 GB RAM)
- **Device GPUs**: 2× NVIDIA Tesla™ C2050 / C2025 (Fermi architecture, `2 × 448` CUDA cores, 4 GB GDDR5 each)
- **Power-Law Fit**: `t(N) = Nᵐ / C`, where `m ≈ 1.000` confirms strict linear complexity and `C` (`photons/s`) is sustained throughput:

| Variant | Architecture & Algorithmic Features | Slope (`m`) | Throughput (`ph/s`) | Wall Time (`10⁹`) | Speedup vs. `S02` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`S02` (Serial)** | Single-core ANSI C baseline on AMD Opteron | `0.972 ± 0.002` | `(2.12 ± 0.08) × 10⁴` | `24,792 s` (~6.89 hr) | `1.0×` |
| **`A01` (Host-Loop)** | 1 photon/thread per launch; outer loop on CPU | `1.00005 ± 0.00001` | `(1.92088 ± 0.00008) × 10⁶` | `521.1 s` (~8.7 min) | `47.6×` |
| **`A02` (In-Kernel)** | Persistent photon loop inside GPU kernel | `0.9996 ± 0.0001` | `(3.095 ± 0.007) × 10⁶` | `320.4 s` (~5.3 min) | `77.4×` |
| **`A06` (Fast-Math)** | `A02` + single-precision hardware SFU (`--use_fast_math`) | `0.9996 ± 0.0001` | `(4.89 ± 0.01) × 10⁶` | `202.8 s` (~3.4 min) | `122.2×` |
| **`A09` (Dual GPU)** | Dual Tesla C2050 + Fast-Math + Pinned Host Memory | `0.9992 ± 0.0001` | **`(11.51 ± 0.04) × 10⁶`** | **`85.4 s` (~1.4 min)** | **`290.3×`** |

---

## 4. Engineering Roadmap Status & Technical Limitations

### ✅ Implemented & Verified
- **Multi-Layer Generalization of the Tumoral GPU Transport Model**: Extends single-layer Basal Cell Carcinoma (`λ = 630 nm`) transport ([Corrales-Machín, 2015](#references--citation); [López-Marín et al., 2016](#references--citation)) to arbitrary multi-layered tissues with Fresnel/Snell boundary crossings and critical-angle total internal reflection (TIR).
- **Persistent In-Loop Photon Respawning & Dynamic Work-Queue (`src/mcml/mcmlgpu.cu`)**: Eliminates SIMT warp lifetime divergence (where 31 lanes in a 32-thread warp previously idled waiting for the longest-lived photon packet to finish). Terminated lanes immediately claim and initialize a fresh photon from an atomic batch queue (`WORK_BATCH_SIZE = 64`).
- **Shared-Memory Hot-Spot Sub-Grid Privatization (`64 × 64` Sub-Grid Cache)**: Eliminates shared-memory spillover on large clinical grids (`nr × nz > 4,096`, such as the `400 × 800` BCC grid). Each thread block caches the high-contention subsurface pencil-beam entry region (`ir < 64, iz < 64`, 32 KB `__shared__` SRAM) and flushes once to global memory on kernel exit.
- **Mixed-Precision Fast-Math GPU Kinematics (`--fp32` Default / `--fp64` Optional)**: Unlocks the 32:1 `FP32`-to-`FP64` throughput advantage and Special Function Units (`__logf`, `__sincosf`, `__fdividef`, `rsqrtf`) on consumer NVIDIA Turing GPUs while accumulating all absorbed, reflected, and transmitted weights in strict 64-bit `FP64` (`double`). Reduces `10⁹`-photon runtime on a single GeForce RTX 2060 from `498.9 s` (`--fp64`) to **`75.1 s`** (`--fp32`).
- **32-Bit Cylindrical Grid Indexing (`int` / `size_t`)**: Removes the legacy 16-bit `short` grid ceiling (`32,767` bins) across `InputStruct`, memory allocators (`mcmlnr.c`), I/O parsers (`mcmlio.c`), and CPU/GPU kernels.
- **Finite-Beam Convolution Post-Processor (`src/conv/` → `build/conv`)**: Standalone C99 CLI tool and library routines that convolve pencil-beam `.mco` impulse responses (`A_rz`, `Rd_r`, `Tt_r`) against finite-radius Gaussian (`TEM₀₀`) and circular flat-top laser beams using adaptive Simpson quadrature (`qtrap`/`qsimp`) and scaled modified Bessel functions `exp(-x)I₀(x)`. Replaces deprecated C89 `gets()` calls with `fgets()` and fixes legacy bugs in `convconv.c` (`A_rzInterp` stub return and missing `Gaussian` branch in `ConvA_rz`).
- **Automated CTest & CI Pipeline**: Unit and integration tests verifying Fresnel optics, Henyey-Greenstein moments, 32-bit grids, Bessel/radial convolution integrals, and CPU/GPU energy conservation.

### ❌ Known Architectural Boundaries (Future Scope)
- **Intra-Step Inter-Layer Branch Divergence**: While persistent respawning eliminates idle-lane lifetime divergence, threads within a warp that simultaneously reside in different layers or take boundary vs. interior branches still serialize conditional paths within a single step.
- **Outer-Radius Global Atomic Fallback (`ir ≥ 64` or `iz ≥ 64`)**: While the `64 × 64` `__shared__` hot-spot cache absorbs the highest-contention pencil-beam entry core, diffuse photons scattering beyond `r ≥ 64·Δr` or `z ≥ 64·Δz` issue global `atomicAdd(double*, double)` transactions to L2 cache.
- **Single-GPU Execution per Process**: Currently dispatches to a single CUDA device; multi-GPU node scaling requires host-side OpenMP/MPI domain partitioning across devices.
- **Planar Multi-Layered Geometry (`z`-Axis Interfaces)**: Tissue interfaces are modeled as horizontal planes (`z = z_k`); curved 3D tumor boundaries require tetrahedral mesh or voxelized ray-tracing.
- **Steady-State (Continuous-Wave) RTE Solver**: Solves the time-independent RTE (`∂I/∂t = 0`); ultrashort pulsed laser dosimetry requires time-of-flight (`Δt`) histogram binning.

---

## Repository Structure

```text
mcml-gpu/
├── .github/workflows/ci.yml  # Multi-compiler CI (GCC/Clang, Release/Debug, CTest)
├── benchmarks/
│   ├── README.md             # Benchmark methodology & scaling definitions
│   └── run_benchmarks.sh     # Automated OpenMP & CUDA benchmark runner (10^9)
├── data/
│   ├── README.md             # Dataset overview & optical parameter table
│   ├── test.mci              # Fast smoke-test configuration (100K photons)
│   ├── validation.mci        # Wang et al. (1995) reference benchmark (1M photons)
│   ├── bcc_tumor_630nm.mci   # Basal Cell Carcinoma PDT model at 630 nm (10^9)
│   └── example.mci           # High-resolution 3-layer run (100M, 800x800 grid)
├── docs/
│   ├── TUTORIAL.md           # Local build, CPU/GPU execution & conv usage guide
│   ├── ARCHITECTURE.md       # RTE physics, diffusion comparison & GPU architecture
│   └── INPUT_FORMAT.md       # Complete .mci input and .mco output specification
├── include/
│   ├── mcml.h                # Core structs, constants & public C/CUDA prototypes
│   ├── conv.h                # Finite-beam convolution structs & prototypes
│   └── extra.h               # Adaptive Simpson quadrature & Bessel I0 prototypes
├── scripts/
│   └── validate_mco.py       # Automated radiative energy conservation verifier
├── src/
│   ├── mcml/
│   │   ├── mcmlmain.c        # CLI entry (--gpu, --cpu, --fp32, --fp64, --threads)
│   │   ├── mcmlgpu.cu        # CUDA kernel (work-queue, 64x64 shared, Mixed FP32)
│   │   ├── mcmlgo.c          # OpenMP CPU Hop-Drop-Spin kernel & thread-safe ran3
│   │   ├── mcmlio.c          # Input parser, validator & ASCII .mco serializer
│   │   └── mcmlnr.c          # Zero-initialized vector/matrix memory allocators
│   └── conv/
│       ├── convmain.c        # Standalone finite-beam convolution CLI (build/conv)
│       ├── convconv.c        # Gaussian & flat-top radial Bessel/Hankel convolution
│       └── extra.c           # Simpson quadrature (qtrap/qsimp) & exp(-x)*I0(x)
├── tests/
│   └── test_physics.c        # C99 unit & physics verification test suite
├── .editorconfig
├── .gitattributes            # Enforces C/CUDA language stats on GitHub Linguist
├── .gitignore                # Excludes local .mco/.dat outputs & build artifacts
├── CITATION.cff              # GitHub citation metadata for published papers
├── CMakeLists.txt            # CMake >= 3.18 build (OpenMP + CUDA sm_75)
├── CONTRIBUTING.md           # Contribution guidelines & engineering standards
└── LICENSE                   # MIT License + scientific attributions
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

# Monte Carlo Multi-Layered (MCML) Media Transport

This repository provides a modernized, HPC-oriented refactoring of the classic MCML code (originally developed by L. Wang, S.L. Jacques, and L. Zheng). The source tree has been reorganized around standard CMake build practices to provide a clean baseline for parallel computing optimizations, including GPU acceleration and large-scale tissue optics simulations.

## Repository Layout

* **`src/mcml/`**: Core Monte Carlo simulation engine.
* **`src/conv/`**: Post-processing convolution utility sources.
* **`include/`**: C header files.
* **`data/`**: Sample `.mci` input configuration files and `.mco` output datasets.
* **`scripts/`**: Analysis scripts and Mathematica notebooks.

## Building from Source

The build system has been migrated to CMake to handle compiler flags and dependencies reliably across different environments. A `Release` build is strongly recommended to enable `-O3` and `-march=native` optimizations.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

This will produce the `mcml` executable in the `build/` directory.

## Usage

Run the simulation by passing an input configuration file (e.g., `.mci`) to the executable. 

```bash
./build/mcml ../data/example.mci
```

The program will parse the input, simulate photon packet trajectories through the defined multi-layered turbid media, and write the distribution results to the corresponding output file specified within the input configuration.

## Development Roadmap

This structural refactor sets the stage for several high-performance computing targets:

- **Multithreading:** Integration of OpenMP to distribute independent photon packet simulations across CPU cores.
- **GPU Offloading:** Porting the core ray-tracing and scattering kernels to CUDA or HIP to leverage massive throughput on modern accelerators.
- **RNG Upgrades:** Replacing legacy pseudo-random number generators with hardware-optimized libraries (e.g., cuRAND or Intel MKL) to prevent pipeline stalling and improve statistical robustness in highly parallel contexts.

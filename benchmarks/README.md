# Benchmark Methodology & Reproducibility

This directory contains the automated benchmarking suite for `mcml-gpu`, configured by default for the **Basal Cell Carcinoma (BCC) Photodynamic Therapy model at $\lambda = 630\text{ nm}$** ($n = 1.37$, $\mu_a = 1.7\text{ cm}^{-1}$, $\mu_s = 125.0\text{ cm}^{-1}$, $g = 0.80$, $d = 0.20\text{ cm}$, $400 \times 800$ cylindrical grid with $\Delta z = \Delta r = 0.0005\text{ cm}$) at **$N = 10^9$ (`1,000,000,000`) photon packets**:

1. **OpenMP Strong Scaling** across `1, 2, 4, 6, 8, N` CPU threads (auto-detecting up to `nproc`).
2. **OpenMP Weak Scaling** ($10^8$ photons per thread by default across `1, 2, 4, 6, 8, N` threads).
3. **CPU (Serial 1T & OpenMP Multi-Thread) vs. NVIDIA CUDA GPU** direct throughput and speedup comparison.

## Running the Benchmark Suite

```bash
chmod +x benchmarks/run_benchmarks.sh

# Run full 10^9-photon benchmark suite (default)
./benchmarks/run_benchmarks.sh

# Optional: override photon counts for a quick hardware check (e.g., 10^6 photons)
MCML_BENCH_PHOTONS=1000000 MCML_WEAK_PHOTONS_PER_THREAD=250000 ./benchmarks/run_benchmarks.sh
```

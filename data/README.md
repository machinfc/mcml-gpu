# Data Directory

This directory contains reference input configuration files (`.mci`) for `mcml-gpu` simulations, spanning quick smoke tests, standard optical benchmarks, and clinical **Photodynamic Therapy (PDT)** tumoral dosimetry models.

## Tracked Reference Inputs (`.mci`)

| File | Physical Model & Reference | Photons | Grid (`nz × nr × na`) | Resolution (`dz × dr`) |
| :--- | :--- | :---: | :---: | :---: |
| [`test.mci`](test.mci) | 3-layer skin phantom (fast CI smoke test) | `100,000` | `40 × 50 × 1` | `0.01 × 0.01 cm` |
| [`validation.mci`](validation.mci) | Standard turbid slab benchmark ([Wang et al., 1995](../README.md#references--citation)) | `1,000,000` | `50 × 100 × 30` | `0.002 × 0.005 cm` |
| [`bcc_tumor_630nm.mci`](bcc_tumor_630nm.mci) | **Basal Cell Carcinoma (BCC) at $\lambda = 630\text{ nm}$** ([López-Marín, Corrales-Machín et al., 2016](../README.md#references--citation); optical parameters from Salomatina et al., 2006: $n=1.37, \mu_a=1.7\text{ cm}^{-1}, \mu_s=125\text{ cm}^{-1}, g=0.80, d=0.20\text{ cm}$) | `1,000,000,000` ($10^9$) | `400 × 800 × 1` | `0.0005 × 0.0005 cm` |
| [`example.mci`](example.mci) | High-resolution 3-layer tumoral tissue model ($10^8$ photons) | `100,000,000` ($10^8$) | `800 × 800 × 1` | `0.0005 × 0.0005 cm` |

See [`../docs/INPUT_FORMAT.md`](../docs/INPUT_FORMAT.md) for the complete `.mci` parameter specification.

## Untracked Local Simulation Outputs (`.mco`)

All simulation outputs (`*.mco`, `*.mco2`, `*.Frzc`, `*.Arzc`, etc.) and local output folders (`results/`, `data/results/`, `data/local/`) are **excluded from Git** via [`.gitignore`](../.gitignore).

To run a simulation and keep your outputs organized locally:

```bash
mkdir -p results
./build/mcml --gpu data/bcc_tumor_630nm.mci
mv bcc_630nm.mco results/
python3 scripts/validate_mco.py results/bcc_630nm.mco
```

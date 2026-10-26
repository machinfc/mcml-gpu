# Input File Format (`.mci`)

This document describes the MCML input file format used to define simulation parameters.

## Overview

Input files use the `.mci` extension and are plain text. Lines starting with `#` are comments. Blank lines are ignored.

## File Structure

```
<file_version>
<number_of_runs>

# --- Run N ---
<output_filename> <format>
<number_of_photons>
<dz> <dr>
<nz> <nr> <na>

<number_of_layers>
<n_above>
<n> <mua> <mus> <g> <d>      # Repeat for each layer
<n_below>
```

## Field Reference

### Global Header

| Field | Type | Description |
|-------|------|-------------|
| `file_version` | float | File format version (use `1.0`) |
| `number_of_runs` | integer | How many simulation runs follow |

### Per-Run Parameters

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `output_filename` | string | — | Name of the output `.mco` file |
| `format` | char | — | `A` for ASCII output, `B` for binary |
| `number_of_photons` | long int | — | Total photon packets to simulate |
| `dz` | double | cm | Grid spacing in the z (depth) direction |
| `dr` | double | cm | Grid spacing in the r (radial) direction |
| `nz` | short | — | Number of grid bins along z |
| `nr` | short | — | Number of grid bins along r |
| `na` | short | — | Number of angular bins (α from 0 to π/2) |
| `number_of_layers` | short | — | Number of tissue layers |

### Layer Parameters

Each layer is defined by a single line with five values:

| Field | Type | Unit | Range | Description |
|-------|------|------|-------|-------------|
| `n` | double | — | > 0 | Refractive index |
| `mua` | double | 1/cm | ≥ 0 | Absorption coefficient |
| `mus` | double | 1/cm | ≥ 0 | Scattering coefficient |
| `g` | double | — | [0, 1] | Anisotropy factor (Henyey-Greenstein) |
| `d` | double | cm | ≥ 0 | Layer thickness |

The **ambient media** (above and below the tissue) are specified with refractive index only.

## Complete Example

```
####
# Monte Carlo simulation of a 3-layer skin model.
# Lengths in cm, mua and mus in 1/cm.
####

1.0                            # File version
1                              # 1 simulation run

### Run 1
test.mco       A               # Output to test.mco in ASCII format
100000                          # 100,000 photon packets
0.01   0.01                     # dz=0.01 cm, dr=0.01 cm
40     50     1                 # 40 depth bins, 50 radial bins, 1 angular bin

3                               # 3 tissue layers
# n      mua    mus    g     d
1.0                             # Air above (n=1.0)
1.37     1      100    0.90  0.1   # Epidermis
1.37     1      10     0     0.1   # Dermis (isotropic scattering)
1.37     2      10     0.70  0.2   # Subcutaneous fat
1.0                             # Air below (n=1.0)
```

## Tips

- **More photons** → smoother distributions but longer runtime. Start with 100K for testing, use 1M+ for publication-quality results.
- **Grid resolution** (`dz`, `dr`) should be fine enough to capture spatial features of interest but not so fine that most bins are empty.
- **Angular bins** (`na`) — use `na=1` if you only need spatially resolved data. Use `na ≥ 10` if you need angular distributions.
- The output `.mco` file is written to the **current working directory**, not the directory containing the `.mci` file.

## Output File Format (`.mco`)

The output file contains:
1. Input parameters (echoed back)
2. **RAT** — Specular reflectance (Rsp), diffuse reflectance (Rd), absorption (A), transmittance (Tt)
3. **1D arrays** — A_l (per-layer absorption), A_z, Rd_r, Rd_a, Tt_r, Tt_a
4. **2D arrays** — A_rz, Rd_ra, Tt_ra

Energy conservation: `Rsp + Rd + A + Tt ≈ 1.0` (within statistical noise).


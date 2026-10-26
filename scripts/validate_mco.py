#!/usr/bin/env python3
"""Validate MCML simulation output (.mco) for physical conservation laws.

Checks that Specular Reflectance (Rsp) + Diffuse Reflectance (Rd) +
Absorption (A) + Total Transmittance (Tt) == 1.0 within Monte Carlo
statistical tolerance.
"""

import math
import sys
from pathlib import Path


def parse_rat(mco_path: Path) -> tuple[float, float, float, float]:
    lines = mco_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("RAT"):
            values = []
            for subline in lines[idx + 1 : idx + 10]:
                stripped = subline.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                token = stripped.split()[0]
                values.append(float(token))
                if len(values) == 4:
                    return values[0], values[1], values[2], values[3]
    raise ValueError(f"Could not find RAT section in {mco_path}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/validate_mco.py <path/to/output.mco> [tolerance]")
        return 1

    mco_path = Path(sys.argv[1])
    tolerance = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.005

    if not mco_path.is_file():
        print(f"[ERROR] Output file not found: {mco_path}", file=sys.stderr)
        return 1

    rsp, rd, a, tt = parse_rat(mco_path)
    total = rsp + rd + a + tt
    error = abs(total - 1.0)

    print(f"--- MCML Output Validation: {mco_path.name} ---")
    print(f"  Specular Reflectance (Rsp) : {rsp:.6f}")
    print(f"  Diffuse Reflectance  (Rd)  : {rd:.6f}")
    print(f"  Absorbed Fraction    (A)   : {a:.6f}")
    print(f"  Total Transmittance  (Tt)  : {tt:.6f}")
    print(f"  ----------------------------------------")
    print(f"  Total Energy Sum           : {total:.6f} (|err| = {error:.6f}, tol = {tolerance:.6f})")

    if math.isnan(total) or error > tolerance:
        print("[FAIL] Energy conservation violated!", file=sys.stderr)
        return 1

    print("[PASS] Radiative energy conservation verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

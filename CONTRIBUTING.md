# Contributing to mcml-gpu

Thank you for your interest in contributing to `mcml-gpu`. We welcome performance optimizations, GPU kernel implementations, bug fixes, and documentation improvements.

## Development Workflow

### 1. Configure & Build (Debug Mode)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DBUILD_TESTING=ON
cmake --build build --parallel
```

### 2. Run the Test Suite

Every pull request must pass all unit and physics conservation tests:

```bash
ctest --test-dir build --output-on-failure --verbose
python3 scripts/validate_mco.py build/test.mco
```

### 3. Run Scaling Benchmarks (For Performance PRs)

If your change modifies the transport kernel (`src/mcml/mcmlgo.c`) or parallel loop (`src/mcml/mcmlmain.c`), run the benchmark suite before and after your change:

```bash
./benchmarks/run_benchmarks.sh
```

## Coding Standards

- **Language Standard**: ISO C99 (`-std=c99 -Wall -Wextra`).
- **Formatting**: 2-space indentation, LF line endings (enforced via [`.editorconfig`](.editorconfig)).
- **Documentation Integrity**: Preserve original copyright headers and algorithmic docstrings when refactoring existing routines.
- **Local Artifacts**: Never commit `.mco` output files, profiling dumps, or binary artifacts.

## Priority Engineering Roadmap

- [ ] Replace `ran3` with stateless counter-based PRNG (`Philox-4x32` or `PCG32`) for deterministic multi-threaded and GPU execution
- [ ] Implement thread-local privatization for near-axis $A_{rz}[0..k][z]$ bins to eliminate atomic cache-line contention at high core counts
- [ ] Add CUDA / HIP device kernels using Structure-of-Arrays (SoA) photon pools
- [ ] Refactor `src/conv/` to replace deprecated C89 `gets()` with `fgets()` and integrate the `conv` target into `CMakeLists.txt`
- [ ] Widen grid indices from `short` (16-bit) to `int32_t`

## License

By contributing to this repository, you agree that your contributions will be licensed under the [MIT License](LICENSE).

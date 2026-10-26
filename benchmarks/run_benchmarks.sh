#!/usr/bin/env bash
# =============================================================================
# Automated Benchmark Suite for MCML-GPU
# Default Physical Model: Basal Cell Carcinoma (BCC) at lambda = 630 nm
#   n = 1.37, mua = 1.7 cm^-1, mus = 125.0 cm^-1, g = 0.80, d = 0.20 cm
#   Grid: dz = 0.0005 cm, dr = 0.0005 cm, nz = 400 (0.2 cm), nr = 800 (0.4 cm)
#   Default Photon Packets: 10^9 (1,000,000,000)
#   (Override photon count via MCML_BENCH_PHOTONS=<N> if needed)
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/benchmarks/results"

mkdir -p "${RESULTS_DIR}"

echo "[INFO] Configuring & building mcml (Release, OpenMP + CUDA)..."
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "${BUILD_DIR}" --parallel > /dev/null

MAX_THREADS=$(nproc 2>/dev/null || echo 12)
THREAD_LIST="1 2 4 6 8 ${MAX_THREADS}"
THREAD_LIST=$(echo "${THREAD_LIST}" | tr ' ' '\n' | awk '!seen[$0]++' | tr '\n' ' ')

# Default benchmark photon count: 10^9 (1,000,000,000)
STRONG_PHOTONS="${MCML_BENCH_PHOTONS:-1000000000}"
BASE_WEAK_PHOTONS="${MCML_WEAK_PHOTONS_PER_THREAD:-100000000}"

generate_bcc_mci() {
  local filepath="$1"
  local outname="$2"
  local photons="$3"
  cat > "${filepath}" <<EOF
1.0
1
${outname} A
${photons}
0.0005 0.0005
400 800 1
1
1.0
1.37 1.7 125.0 0.80 0.20
1.0
EOF
}

extract_kernel_sec() {
  grep "Wall/User time:" | awk '{print $3}'
}

echo ""
echo "============================================================================"
echo " 1. OpenMP Strong Scaling — Basal Cell Carcinoma 630 nm (${STRONG_PHOTONS} Photons)"
echo "============================================================================"
printf "%-10s | %-12s | %-18s | %-10s | %-10s\n" "Threads" "Time (s)" "Throughput (ph/s)" "Speedup" "Efficiency"
printf "%-10s-+-%-12s-+-%-18s-+-%-10s-+-%-10s\n" "----------" "------------" "------------------" "----------" "----------"

STRONG_MCI="${RESULTS_DIR}/strong.mci"
generate_bcc_mci "${STRONG_MCI}" "benchmarks/results/strong.mco" "${STRONG_PHOTONS}"

T1_TIME=""
TMAX_TIME=""
for threads in ${THREAD_LIST}; do
  elapsed_s=$("${BUILD_DIR}/mcml" --cpu --threads "${threads}" "${STRONG_MCI}" | extract_kernel_sec)
  if [[ -z "${T1_TIME}" ]]; then
    T1_TIME="${elapsed_s}"
  fi
  TMAX_TIME="${elapsed_s}"
  throughput=$(awk "BEGIN {printf \"%.0f\", ${STRONG_PHOTONS} / ${elapsed_s}}")
  speedup=$(awk "BEGIN {printf \"%.2fx\", ${T1_TIME} / ${elapsed_s}}")
  efficiency=$(awk "BEGIN {printf \"%.1f%%\", (${T1_TIME} / (${elapsed_s} * ${threads})) * 100.0}")

  printf "%-10s | %-12s | %-18s | %-10s | %-10s\n" "${threads}" "${elapsed_s}" "${throughput}" "${speedup}" "${efficiency}"
done

echo ""
echo "============================================================================"
echo " 2. OpenMP Weak Scaling — Basal Cell Carcinoma 630 nm (${BASE_WEAK_PHOTONS} Photons/Thread)"
echo "============================================================================"
printf "%-10s | %-14s | %-12s | %-18s | %-15s\n" "Threads" "Total Photons" "Time (s)" "Throughput (ph/s)" "Weak Efficiency"
printf "%-10s-+-%-14s-+-%-12s-+-%-18s-+-%-15s\n" "----------" "--------------" "------------" "------------------" "---------------"

W1_TIME=""
for threads in ${THREAD_LIST}; do
  total_photons=$((BASE_WEAK_PHOTONS * threads))
  WEAK_MCI="${RESULTS_DIR}/weak_${threads}.mci"
  generate_bcc_mci "${WEAK_MCI}" "benchmarks/results/weak_${threads}.mco" "${total_photons}"

  elapsed_s=$("${BUILD_DIR}/mcml" --cpu --threads "${threads}" "${WEAK_MCI}" | extract_kernel_sec)
  if [[ -z "${W1_TIME}" ]]; then
    W1_TIME="${elapsed_s}"
  fi
  throughput=$(awk "BEGIN {printf \"%.0f\", ${total_photons} / ${elapsed_s}}")
  weak_eff=$(awk "BEGIN {printf \"%.1f%%\", (${W1_TIME} / ${elapsed_s}) * 100.0}")

  printf "%-10s | %-14s | %-12s | %-18s | %-15s\n" "${threads}" "${total_photons}" "${elapsed_s}" "${throughput}" "${weak_eff}"
done

echo ""
echo "============================================================================"
echo " 3. CPU (Serial & OpenMP ${MAX_THREADS}T) vs. NVIDIA CUDA GPU (${STRONG_PHOTONS} Photons)"
echo "============================================================================"
printf "%-30s | %-12s | %-18s | %-12s\n" "Backend" "Time (s)" "Throughput (ph/s)" "Speedup vs 1T"
printf "%-30s-+-%-12s-+-%-18s-+-%-12s\n" "------------------------------" "------------" "------------------" "------------"

CPU1_TP=$(awk "BEGIN {printf \"%.0f\", ${STRONG_PHOTONS} / ${T1_TIME}}")
printf "%-30s | %-12s | %-18s | %-12s\n" "CPU Serial (1 Thread)" "${T1_TIME}" "${CPU1_TP}" "1.00x"

CPUN_TP=$(awk "BEGIN {printf \"%.0f\", ${STRONG_PHOTONS} / ${TMAX_TIME}}")
CPUN_SP=$(awk "BEGIN {printf \"%.2fx\", ${T1_TIME} / ${TMAX_TIME}}")
printf "%-30s | %-12s | %-18s | %-12s\n" "CPU OpenMP (${MAX_THREADS} Threads)" "${TMAX_TIME}" "${CPUN_TP}" "${CPUN_SP}"

GPU_MCI="${RESULTS_DIR}/gpu.mci"
generate_bcc_mci "${GPU_MCI}" "benchmarks/results/gpu.mco" "${STRONG_PHOTONS}"

GPU_OUT=$("${BUILD_DIR}/mcml" --gpu "${GPU_MCI}")
if echo "${GPU_OUT}" | grep -q "NVIDIA CUDA GPU"; then
  GPU_TIME=$(echo "${GPU_OUT}" | extract_kernel_sec)
  GPU_TP=$(awk "BEGIN {printf \"%.0f\", ${STRONG_PHOTONS} / ${GPU_TIME}}")
  GPU_SP=$(awk "BEGIN {printf \"%.2fx\", ${T1_TIME} / ${GPU_TIME}}")
  printf "%-30s | %-12s | %-18s | %-12s\n" "NVIDIA CUDA GPU" "${GPU_TIME}" "${GPU_TP}" "${GPU_SP}"
fi

echo ""
echo "[OK] Local benchmark artifacts stored in ${RESULTS_DIR} (ignored by Git)."

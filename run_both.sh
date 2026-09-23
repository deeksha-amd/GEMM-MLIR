#!/usr/bin/env bash
# Run PACE TPP GEMM and the MLIR lowering beside it. Dumps stay separate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "########## 1/2 PACE TPP (libXSMM vdpbf16ps) ##########"
# Inspect dumps kernels; skip if you only want MLIR: PACE_INSPECT=0
./setup_and_run.sh

echo
echo "########## 2/2 MLIR linalg.matmul (step dumps in mlir/out/) ##########"
chmod +x mlir/run_steps.sh mlir/lower_amx.sh mlir/run_portable.sh
./mlir/run_steps.sh
./mlir/lower_amx.sh || true

echo
echo "Compare (mnemonics, not GFLOPS):"
echo "  recommended IR: cat mlir/out/22_portable_proof.txt"
echo "  PACE:           grep vdpbf16ps kernel_dump/*br3*.mxm.s | head"
echo "  2x2:            cat mlir/out/09_isa_hits.txt"
echo "  contract:       cat mlir/out/13_contract_isa.txt"
echo "  decoder only:   cat mlir/out/18_dpbf16_isa.txt"
echo "  GFLOP/s:        cat mlir/out/flops.txt   (./mlir/bench_flops.sh)"
echo "Docs:    mlir/README.md   mlir/STEPS.md"

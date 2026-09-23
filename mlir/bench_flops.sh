#!/usr/bin/env bash
# FLOPs / GFLOP/s table: PACE vs matmul vs vector.contract vs portable MLIR.
set -euo pipefail
cd "$(dirname "$0")"
for _d in ${MLIR_BIN:-} /opt/rocm/llvm/bin /opt/rocm-*/llvm/bin; do
  [[ -x "$_d/mlir-opt" && -x "$_d/mlir-runner" ]] || continue
  export PATH="$_d:${PATH}"
  export MLIR_BIN="${MLIR_BIN:-$_d}"
  break
done
exec python3 ./bench_flops.py "$@"

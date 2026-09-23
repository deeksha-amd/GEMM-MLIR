#!/usr/bin/env bash
# Layer 1–2 only: linalg.matmul + tile. No ISA in the IR.
# Usage: ./run_portable.sh
set -euo pipefail
for _d in ${MLIR_BIN:-} /opt/rocm/llvm/bin /opt/rocm-*/llvm/bin; do
  [[ -x "$_d/mlir-opt" ]] || continue
  export PATH="$_d:${PATH}"
  break
done
cd "$(dirname "$0")"
mkdir -p out
command -v mlir-opt >/dev/null || { echo "missing mlir-opt"; exit 1; }

echo "=== layer 1: source (must contain linalg.matmul, must NOT contain vdpbf16/avx) ==="
if grep -E 'vdpbf16|call_intrinsic|avx512' portable_matmul.mlir; then
  echo "FAIL: portable IR named an ISA" >&2
  exit 1
fi
grep -n 'linalg.matmul' portable_matmul.mlir

echo
echo "=== layer 2: --transform-interpreter (tiles around linalg.matmul) ==="
if ! mlir-opt portable_matmul.mlir --transform-interpreter --canonicalize \
     > out/21_portable_tiled.mlir 2>out/21_portable_tile_err.txt; then
  echo "transform-interpreter failed; affine-loop-tile fallback" >&2
  cat out/21_portable_tile_err.txt >&2 || true
  mlir-opt portable_matmul.mlir \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-affine-loops \
    --affine-loop-tile="tile-sizes=8,8,8" \
    > out/21_portable_tiled.mlir
fi
grep -n -E 'linalg.matmul|scf.for|affine.for' out/21_portable_tiled.mlir | head -20
echo
echo "Wrote out/21_portable_tiled.mlir"
echo "Layer 3 (backend) is not in this IR — PACE, IREE, or CPU LLVM."
echo "See DESIGN.md"

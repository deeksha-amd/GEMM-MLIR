#!/usr/bin/env bash
# Easier path: packaged LLVM 22 in Docker (no source build).
# Proves linalg.matmul -> pack -> tile -> vectorize -> vector.contract
#          -> x86vector.avx512.dot -> vdpbf16ps
#
#   ./chain/run.sh
#   docker build -t pace-learn-mlir:llvm22 -f chain/Dockerfile chain/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-pace-learn-mlir:llvm22}"
MLIR_BIN="${MLIR_BIN:-/usr/lib/llvm-22/bin}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE from chain/Dockerfile"
  docker build -t "$IMAGE" -f "$ROOT/chain/Dockerfile" "$ROOT/chain"
fi

docker run --rm \
  --security-opt seccomp=unconfined \
  -v "$ROOT":/work \
  -w /work \
  -e MLIR_BIN="$MLIR_BIN" \
  "$IMAGE" \
  bash /work/chain/run_in_container.sh

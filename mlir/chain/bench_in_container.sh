#!/usr/bin/env bash
# Time the 12x32 pack->dot chain at 128x4096x4096. Invoked inside pace-learn-mlir:llvm22:
#   docker run --rm --security-opt seccomp=unconfined -v "$PWD":/work -w /work \
#     pace-learn-mlir:llvm22 bash /work/chain/bench_in_container.sh
set -euo pipefail
export PATH="${MLIR_BIN:-/usr/lib/llvm-22/bin}:$PATH"
LIB=/usr/lib/llvm-22/lib
OUT=/work/chain/out
mkdir -p "$OUT"
ln -sf libmlir_runner_utils.so.22.1 "$LIB/libmlir_runner_utils.so"
ln -sf libmlir_c_runner_utils.so.22.1 "$LIB/libmlir_c_runner_utils.so"

# 12x32 register tile + B panels. Flatten transfers instead of vector-to-scf
# so the 24 C zmms stay in registers (scf lowering spilled them to ~90 GFLOP/s).
run_prod12() {
  local src=/work/chain/bench_prod_12x32.mlir
  echo
  echo "======== prod12 128x4096x4096 (MR=12 NR=32, M padded 132) ========"
  mlir-opt "$src" --transform-interpreter --test-transform-dialect-erase-schedule \
    --canonicalize --cse -o "$OUT/prod12_sched.mlir"
  echo "  contracts=$(grep -c vector.contract "$OUT/prod12_sched.mlir" || true)"
  python3 /work/chain/reshape_vnni_contract.py \
    < "$OUT/prod12_sched.mlir" > "$OUT/prod12_reshaped.mlir"
  python3 - "$OUT/prod12_reshaped.mlir" << 'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1]).read_text()
if "transform.with_named_sequence" not in src.split("func.func")[0]:
    if "module attributes {" in src:
        src = src.replace("module attributes {", "module attributes {transform.with_named_sequence, ", 1)
    else:
        src = src.replace("module {", "module attributes {transform.with_named_sequence} {", 1)
seq = """
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %fn = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fn {
      transform.apply_patterns.x86vector.vector_contract_to_packed_type_dot_product
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
"""
src = src.rstrip()
if src.endswith("}"):
    src = src[:-1] + seq + "}\n"
Path(sys.argv[1]).write_text(src)
PY
  mlir-opt "$OUT/prod12_reshaped.mlir" --transform-interpreter \
    --test-transform-dialect-erase-schedule --canonicalize --cse \
    -o "$OUT/prod12_dots.mlir"
  echo "  dots=$(grep -c x86vector.avx512.dot "$OUT/prod12_dots.mlir" || true)"
  mlir-opt "$OUT/prod12_dots.mlir" \
    --canonicalize --cse \
    --one-shot-bufferize=bufferize-function-boundaries \
    --canonicalize --cse \
    --test-vector-transfer-flatten-patterns \
    --canonicalize --cse \
    --convert-linalg-to-loops \
    --canonicalize --cse \
    --loop-invariant-code-motion \
    --convert-scf-to-cf \
    --expand-strided-metadata --lower-affine \
    --convert-vector-to-llvm=enable-x86vector \
    --convert-arith-to-llvm --convert-ub-to-llvm \
    --finalize-memref-to-llvm --convert-func-to-llvm \
    --convert-cf-to-llvm --convert-index-to-llvm --reconcile-unrealized-casts \
    -o "$OUT/prod12.llvm.mlir"
  echo "  dpbf16ps=$(grep -c dpbf16ps "$OUT/prod12.llvm.mlir" || true)"
  local nums
  set +e
  nums=$(ulimit -s unlimited; mlir-runner --O3 -e main --entry-point-result=void \
    --shared-libs="$LIB/libmlir_c_runner_utils.so,$LIB/libmlir_runner_utils.so" \
    "$OUT/prod12.llvm.mlir")
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    echo "  mlir-runner failed with status $status"
    return 0
  fi
  echo "$nums"
  python3 - "$nums" "$OUT/prod12_flops.txt" << 'PY'
import sys
secs, chk = [float(x) for x in sys.argv[1].split()]
M, N, K, reps = 128, 4096, 4096, 1
gflops = 2 * M * N * K * reps / secs / 1e9
expect = float(K * (1 + reps))
ok = "OK" if abs(chk - expect) < 1 else "CHECK FAIL"
print(f"  FLOPs/gemm={2*M*N*K}  time={secs:.4f}s  GFLOP/s={gflops:.2f}  C[0]={chk:.0f} expect={expect:.0f}  {ok}")
open(sys.argv[2], "w").write(
    f"prod12x32 M={M} N={N} K={K} secs={secs} gflops={gflops} chk={chk} {ok}\n")
PY
}

run_prod12

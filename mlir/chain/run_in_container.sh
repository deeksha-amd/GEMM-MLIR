#!/usr/bin/env bash
# Invoked inside pace-learn-mlir:llvm22.
# Proves: linalg.matmul -> pack -> tile -> vectorize -> vector.contract
#          -> x86vector.avx512.dot -> vdpbf16ps
set -euo pipefail
BIN="${MLIR_BIN:-/usr/lib/llvm-22/bin}"
export PATH="$BIN:$PATH"
OUT=/work/chain/out
LIB=/usr/lib/llvm-22/lib
mkdir -p "$OUT"
LOG="$OUT/run.log"
: > "$LOG"
log() { printf '\n======== %s ========\n' "$*" | tee -a "$LOG"; }

echo "mlir-opt $(mlir-opt --version | head -1)" | tee "$LOG"

ln -sf libmlir_runner_utils.so.22.1 "$LIB/libmlir_runner_utils.so"
ln -sf libmlir_c_runner_utils.so.22.1 "$LIB/libmlir_c_runner_utils.so"

LOWER_TO_LLVM=(
  --test-transform-dialect-erase-schedule
  --loop-invariant-code-motion
  --loop-invariant-subset-hoisting
  --canonicalize --cse
  --one-shot-bufferize=bufferize-function-boundaries
  --canonicalize --cse
  --convert-vector-to-scf
  --convert-linalg-to-loops
  --canonicalize --cse
  --convert-scf-to-cf
  --expand-strided-metadata --lower-affine
  --convert-vector-to-llvm=enable-x86vector
  --convert-arith-to-llvm
  --convert-ub-to-llvm
  --finalize-memref-to-llvm
  --convert-func-to-llvm
  --convert-cf-to-llvm
  --convert-index-to-llvm
  --reconcile-unrealized-casts
)

# --- A. x86vector.avx512.dot parses and becomes vdpbf16ps ---
log "A. x86vector.avx512.dot -> vdpbf16ps"
cat > "$OUT/A_dot.mlir" << 'MLIR'
func.func @dot(%a: vector<32xbf16>, %b: vector<32xbf16>, %c: vector<16xf32>) -> vector<16xf32> {
  %0 = x86vector.avx512.dot %c, %a, %b : vector<32xbf16> -> vector<16xf32>
  return %0 : vector<16xf32>
}
MLIR
mlir-opt "$OUT/A_dot.mlir" --convert-vector-to-llvm=enable-x86vector \
  --convert-func-to-llvm -o "$OUT/A_dot.llvm.mlir"
mlir-translate --mlir-to-llvmir "$OUT/A_dot.llvm.mlir" -o "$OUT/A_dot.ll"
llc -O3 -mcpu=znver4 -mattr=+avx512f,+avx512bf16 "$OUT/A_dot.ll" -o "$OUT/A_dot.s"
grep -n "dpbf16ps\|vdpbf16ps" "$OUT/A_dot.llvm.mlir" "$OUT/A_dot.s" | tee -a "$LOG"

# --- B. vector.contract (VNNI) -> x86vector.dot -> vdpbf16ps ---
log "B. vector.contract -> x86vector.avx512.dot -> vdpbf16ps"
mlir-opt /work/chain/01_contract_to_dot.mlir --transform-interpreter \
  --canonicalize -o "$OUT/B_dot.mlir"
mlir-opt "$OUT/B_dot.mlir" --test-transform-dialect-erase-schedule \
  --convert-vector-to-llvm=enable-x86vector --convert-ub-to-llvm \
  --convert-func-to-llvm --reconcile-unrealized-casts -o "$OUT/B_dot.llvm.mlir"
mlir-translate --mlir-to-llvmir "$OUT/B_dot.llvm.mlir" -o "$OUT/B_dot.ll"
llc -O3 -mcpu=znver4 -mattr=+avx512f,+avx512bf16 "$OUT/B_dot.ll" -o "$OUT/B_dot.s"
grep -n "x86vector.avx512.dot\|dpbf16ps\|vdpbf16ps" \
  "$OUT/B_dot.mlir" "$OUT/B_dot.llvm.mlir" "$OUT/B_dot.s" | tee -a "$LOG"

# --- C. pack only, then full schedule on linalg.matmul ---
log "C1. pack K into VNNI pairs (packed_sizes = [0,0,2])"
cat > "$OUT/C_pack_only.mlir" << 'MLIR'
module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<4x32xbf16>,
                    %B: tensor<32x16xbf16>,
                    %C: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %0 = linalg.matmul
        ins(%A, %B : tensor<4x32xbf16>, tensor<32x16xbf16>)
        outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %mm packed_sizes = [0, 0, 2]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}
MLIR
mlir-opt "$OUT/C_pack_only.mlir" --transform-interpreter --canonicalize \
  -o "$OUT/C1_packed.mlir"
grep -n "linalg.pack\|inner_tiles" "$OUT/C1_packed.mlir" | tee -a "$LOG"

log "C2. pack + tile + vectorize + packed-dot"
mlir-opt /work/chain/02_matmul_pack.mlir --transform-interpreter \
  --canonicalize -o "$OUT/C2_scheduled.mlir"
grep -n "x86vector.avx512.dot\|vector.contract\|linalg.pack" \
  "$OUT/C2_scheduled.mlir" | tee -a "$LOG"

log "C3. scheduled IR -> llvm -> asm"
mlir-opt "$OUT/C2_scheduled.mlir" "${LOWER_TO_LLVM[@]}" -o "$OUT/C3.llvm.mlir"
mlir-translate --mlir-to-llvmir "$OUT/C3.llvm.mlir" -o "$OUT/C3.ll"
llc -O3 -mcpu=znver4 -mattr=+avx512f,+avx512bf16 "$OUT/C3.ll" -o "$OUT/C3.s"
echo "dpbf16ps in llvm dialect: $(grep -c dpbf16ps "$OUT/C3.llvm.mlir")" | tee -a "$LOG"
echo "vdpbf16ps in asm:         $(grep -c vdpbf16ps "$OUT/C3.s")" | tee -a "$LOG"
grep -n "vdpbf16ps" "$OUT/C3.s" | tee -a "$LOG"

# --- D. run it: all-ones 4x32 * 32x16, C must be 32 ---
log "D. mlir-runner correctness (expect 32 32 32)"
mlir-opt /work/chain/03_matmul_run.mlir --transform-interpreter \
  --canonicalize -o "$OUT/D_scheduled.mlir"
mlir-opt "$OUT/D_scheduled.mlir" "${LOWER_TO_LLVM[@]}" -o "$OUT/D.llvm.mlir"
mlir-runner --O3 -e main --entry-point-result=void \
  --shared-libs="$LIB/libmlir_c_runner_utils.so,$LIB/libmlir_runner_utils.so" \
  "$OUT/D.llvm.mlir" | tee "$OUT/D_runner.txt" | tee -a "$LOG"

log "DONE"
echo "Artifacts in /work/chain/out" | tee -a "$LOG"

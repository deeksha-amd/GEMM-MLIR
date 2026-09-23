// Runnable like-to-like check for the automatic chain:
//   linalg.matmul (bf16*bf16->f32)
//   -> pack K into VNNI pairs of 2
//   -> tile [1,16,1]
//   -> vectorize
//   -> fold arith.extf into vector.contract
//   -> x86vector.avx512.dot
//   -> vdpbf16ps
//
// All-ones 4x32 * 32x16, so every C[i,j] must be 32.0.
module attributes {transform.with_named_sequence} {
  func.func private @printF64(f64)
  func.func private @printNewline()

  func.func @matmul(%A: tensor<4x32xbf16>,
                    %B: tensor<32x16xbf16>,
                    %C: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %0 = linalg.matmul
        ins(%A, %B : tensor<4x32xbf16>, tensor<32x16xbf16>)
        outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c15 = arith.constant 15 : index
    %one = arith.constant 1.0 : bf16
    %zf = arith.constant 0.0 : f32

    %Ae = tensor.empty() : tensor<4x32xbf16>
    %Be = tensor.empty() : tensor<32x16xbf16>
    %Ce = tensor.empty() : tensor<4x16xf32>
    %A = linalg.fill ins(%one : bf16) outs(%Ae : tensor<4x32xbf16>) -> tensor<4x32xbf16>
    %B = linalg.fill ins(%one : bf16) outs(%Be : tensor<32x16xbf16>) -> tensor<32x16xbf16>
    %C0 = linalg.fill ins(%zf : f32) outs(%Ce : tensor<4x16xf32>) -> tensor<4x16xf32>

    %C = func.call @matmul(%A, %B, %C0)
        : (tensor<4x32xbf16>, tensor<32x16xbf16>, tensor<4x16xf32>) -> tensor<4x16xf32>

    %v00 = tensor.extract %C[%c0, %c0] : tensor<4x16xf32>
    %v0l = tensor.extract %C[%c0, %c15] : tensor<4x16xf32>
    %v30 = tensor.extract %C[%c3, %c0] : tensor<4x16xf32>
    %d00 = arith.extf %v00 : f32 to f64
    %d0l = arith.extf %v0l : f32 to f64
    %d30 = arith.extf %v30 : f32 to f64
    func.call @printF64(%d00) : (f64) -> ()
    func.call @printNewline() : () -> ()
    func.call @printF64(%d0l) : (f64) -> ()
    func.call @printNewline() : () -> ()
    func.call @printF64(%d30) : (f64) -> ()
    func.call @printNewline() : () -> ()
    return
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %mm packed_sizes = [0, 0, 2]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    %gx = transform.structured.interchange %packed iterator_interchange = [1, 0, 2, 3]
        : (!transform.op<"linalg.generic">) -> !transform.any_op
    %l1, %n, %m, %k = transform.structured.tile_using_for %gx
        tile_sizes [16, 1, 1]
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op,
            !transform.any_op, !transform.any_op)
    %fn = transform.get_parent_op %l1 {op_name = "func.func"}
        : (!transform.any_op) -> !transform.any_op
    %vfn = transform.structured.vectorize_children_and_apply_patterns %fn
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %vfn {
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.x86vector.vector_contract_to_packed_type_dot_product
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %loops = transform.structured.match ops{["scf.for"]} in %vfn
        : (!transform.any_op) -> !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %loops : !transform.any_op
    transform.yield
  }
}

module attributes {transform.with_named_sequence} {
  func.func private @rtclock() -> f64
  func.func private @printF64(f64)
  func.func private @printNewline()
  func.func @matmul(%A: tensor<4x32xbf16>, %B: tensor<32x16xbf16>, %C: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<4x32xbf16>, tensor<32x16xbf16>)
                       outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %reps = arith.constant 100000 : index
    %one = arith.constant 1.0 : bf16
    %zf = arith.constant 0.0 : f32
    %Ae = tensor.empty() : tensor<4x32xbf16>
    %Be = tensor.empty() : tensor<32x16xbf16>
    %Ce = tensor.empty() : tensor<4x16xf32>
    %A = linalg.fill ins(%one : bf16) outs(%Ae : tensor<4x32xbf16>) -> tensor<4x32xbf16>
    %B = linalg.fill ins(%one : bf16) outs(%Be : tensor<32x16xbf16>) -> tensor<32x16xbf16>
    %C0 = linalg.fill ins(%zf : f32) outs(%Ce : tensor<4x16xf32>) -> tensor<4x16xf32>
    %Cw = func.call @matmul(%A, %B, %C0)
        : (tensor<4x32xbf16>, tensor<32x16xbf16>, tensor<4x16xf32>) -> tensor<4x16xf32>
    %t0 = func.call @rtclock() : () -> f64
    %Cr = scf.for %r = %c0 to %reps step %c1
        iter_args(%acc = %Cw) -> (tensor<4x16xf32>) {
      %t = func.call @matmul(%A, %B, %acc)
          : (tensor<4x32xbf16>, tensor<32x16xbf16>, tensor<4x16xf32>) -> tensor<4x16xf32>
      scf.yield %t : tensor<4x16xf32>
    }
    %t1 = func.call @rtclock() : () -> f64
    %dt = arith.subf %t1, %t0 : f64
    func.call @printF64(%dt) : (f64) -> ()
    func.call @printNewline() : () -> ()
    %v = tensor.extract %Cr[%c0, %c0] : tensor<4x16xf32>
    %vf = arith.extf %v : f32 to f64
    func.call @printF64(%vf) : (f64) -> ()
    func.call @printNewline() : () -> ()
    return
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %mm packed_sizes = [0, 0, 2]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    %gx = transform.structured.interchange %packed iterator_interchange = [1, 0, 2, 3]
        : (!transform.op<"linalg.generic">) -> !transform.any_op
    %l1, %n, %m, %k = transform.structured.tile_using_for %gx tile_sizes [16, 1, 1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
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

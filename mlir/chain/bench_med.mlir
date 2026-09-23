#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func private @rtclock() -> f64
  func.func private @printF64(f64)
  func.func private @printNewline()
  func.func @matmul(%A: tensor<128x128x2xbf16>,
                    %B: tensor<128x256x2xbf16>,
                    %C: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2],
                         iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
        ins(%A, %B : tensor<128x128x2xbf16>, tensor<128x256x2xbf16>)
        outs(%C : tensor<128x256xf32>) {
    ^bb0(%a: bf16, %b: bf16, %c: f32):
      %ae = arith.extf %a : bf16 to f32
      %be = arith.extf %b : bf16 to f32
      %m = arith.mulf %ae, %be : f32
      %r = arith.addf %c, %m : f32
      linalg.yield %r : f32
    } -> tensor<128x256xf32>
    return %0 : tensor<128x256xf32>
  }
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %reps = arith.constant 10 : index
    %one = arith.constant 1.0 : bf16
    %zf = arith.constant 0.0 : f32
    %Ae = tensor.empty() : tensor<128x128x2xbf16>
    %Be = tensor.empty() : tensor<128x256x2xbf16>
    %Ce = tensor.empty() : tensor<128x256xf32>
    %A = linalg.fill ins(%one : bf16) outs(%Ae : tensor<128x128x2xbf16>) -> tensor<128x128x2xbf16>
    %B = linalg.fill ins(%one : bf16) outs(%Be : tensor<128x256x2xbf16>) -> tensor<128x256x2xbf16>
    %C0 = linalg.fill ins(%zf : f32) outs(%Ce : tensor<128x256xf32>) -> tensor<128x256xf32>
    %Cw = func.call @matmul(%A, %B, %C0)
        : (tensor<128x128x2xbf16>, tensor<128x256x2xbf16>, tensor<128x256xf32>) -> tensor<128x256xf32>
    %t0 = func.call @rtclock() : () -> f64
    %Cr = scf.for %r = %c0 to %reps step %c1
        iter_args(%acc = %Cw) -> (tensor<128x256xf32>) {
      %t = func.call @matmul(%A, %B, %acc)
          : (tensor<128x128x2xbf16>, tensor<128x256x2xbf16>, tensor<128x256xf32>) -> tensor<128x256xf32>
      scf.yield %t : tensor<128x256xf32>
    }
    %t1 = func.call @rtclock() : () -> f64
    %dt = arith.subf %t1, %t0 : f64
    func.call @printF64(%dt) : (f64) -> ()
    func.call @printNewline() : () -> ()
    %v = tensor.extract %Cr[%c0, %c0] : tensor<128x256xf32>
    %vf = arith.extf %v : f32 to f64
    func.call @printF64(%vf) : (f64) -> ()
    func.call @printNewline() : () -> ()
    return
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %gx = transform.structured.interchange %g iterator_interchange = [1, 0, 2, 3]
        : (!transform.any_op) -> !transform.any_op
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

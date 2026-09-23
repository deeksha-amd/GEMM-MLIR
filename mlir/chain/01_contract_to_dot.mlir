// LLVM 22: vector.contract (VNNI bf16, f32 acc) -> x86vector.avx512.dot
// Shape is the upstream test: M=1, N=16, K=1, VNNI=2.
module attributes {transform.with_named_sequence} {
  func.func @contract_vnni(%a: vector<1x1x2xbf16>,
                           %b: vector<1x16x2xbf16>,
                           %c: vector<1x16xf32>) -> vector<1x16xf32> {
    %0 = vector.contract {
      indexing_maps = [affine_map<(d4, d1, d2, d3) -> (d1, d3, d4)>,
                       affine_map<(d4, d1, d2, d3) -> (d3, d2, d4)>,
                       affine_map<(d4, d1, d2, d3) -> (d1, d2)>],
      iterator_types = ["reduction", "parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %a, %b, %c : vector<1x1x2xbf16>, vector<1x16x2xbf16> into vector<1x16xf32>
    return %0 : vector<1x16xf32>
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %fn = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fn {
      transform.apply_patterns.x86vector.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

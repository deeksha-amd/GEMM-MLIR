// 12x32 register tile + sequential-K B panels, transform dialect only.
// No handwritten ukernel.
//
//   linalg.matmul
//     -> pack MR=12, NR=32, VNNI K=2
//     -> pack_transpose B to [N/32, K/2, 32, 2]   (K walks 128B lines)
//     -> interchange N_o outer
//     -> tile [1, 1, 1, 1, 16] then unroll 12 x 2 into K
//     -> vectorize, hoist 24 C vectors
//     -> reshape 6D unit-M contract -> 4D VNNI (see reshape_vnni_contract.py)
//     -> packed-dot -> 24 x vdpbf16ps
module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<24x32xbf16>,
                    %B: tensor<32x64xbf16>,
                    %C: tensor<24x64xf32>) -> tensor<24x64xf32> {
    %0 = linalg.matmul
        ins(%A, %B : tensor<24x32xbf16>, tensor<32x64xbf16>)
        outs(%C : tensor<24x64xf32>) -> tensor<24x64xf32>
    return %0 : tensor<24x64xf32>
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %mm packed_sizes = [12, 32, 2]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    %packs = transform.structured.match ops{["linalg.pack"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %ha, %hb, %hc = transform.split_handle %packs
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.pack_transpose %hb
        with_compute_op(%packed) outer_perm = [1, 0]
        : (!transform.any_op, !transform.op<"linalg.generic">)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %gens = transform.structured.match ops{["linalg.generic"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %gx = transform.structured.interchange %gens
        iterator_interchange = [1, 0, 2, 3, 4, 5]
        : (!transform.any_op) -> !transform.any_op
    %l1, %no, %mo, %ko, %mi, %ni = transform.structured.tile_using_for %gx
        tile_sizes [1, 1, 1, 1, 16]
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op,
            !transform.any_op, !transform.any_op, !transform.any_op)
    transform.loop.unroll %ni {factor = 2} : !transform.any_op
    transform.loop.unroll %mi {factor = 12} : !transform.any_op
    %fn = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %vfn = transform.structured.vectorize_children_and_apply_patterns %fn
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %vfn {
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %loops = transform.structured.match ops{["scf.for"]} in %vfn
        : (!transform.any_op) -> !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %loops : !transform.any_op
    transform.yield
  }
}

// Production-shaped GEMM: the math never names an ISA.
// C = A @ B   (same as PACE Linear with A=X, B=W.T)
//
// Layer 1 — @matmul: stay here when a new CPU/GPU appears.
// Layer 2 — tile schedule below: loops wrap the SAME linalg.matmul.
// Layer 3 — backend not in this file (CPU LLVM, IREE, Triton, PACE).
//
// dpbf16ps.mlir hardcodes an x86 intrinsic. That is not this design.
//
//   mlir-opt portable_matmul.mlir --transform-interpreter --canonicalize

module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<16x16xf32>, %B: tensor<16x16xf32>,
                    %C: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<16x16xf32>, tensor<16x16xf32>)
                       outs(%C : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %tiled, %ii, %jj, %kk = transform.structured.tile_using_for %mm
        tile_sizes [8, 8, 8]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                  !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// portable_matmul.mlir style at production size: linalg.matmul + tile schedule.
// Same C = A @ B  128x4096x4096 bf16. Tiles [32,64,64] divide M,N,K.
// Inner op stays linalg.matmul (no vdpbf16ps in source).
// Element type stays bf16: linalg.matmul bf16 parses and lowers on AMD LLVM 20.

module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<128x4096xbf16>, %B: tensor<4096x4096xbf16>,
                    %C: tensor<128x4096xbf16>) -> tensor<128x4096xbf16> {
    %0 = linalg.matmul ins(%A, %B : tensor<128x4096xbf16>, tensor<4096x4096xbf16>)
                       outs(%C : tensor<128x4096xbf16>) -> tensor<128x4096xbf16>
    return %0 : tensor<128x4096xbf16>
  }

  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : bf16
    %z = arith.constant 0.0 : bf16
    %Ae = tensor.empty() : tensor<128x4096xbf16>
    %Be = tensor.empty() : tensor<4096x4096xbf16>
    %Ce = tensor.empty() : tensor<128x4096xbf16>
    %A = linalg.fill ins(%one : bf16) outs(%Ae : tensor<128x4096xbf16>) -> tensor<128x4096xbf16>
    %B = linalg.fill ins(%one : bf16) outs(%Be : tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
    %C0 = linalg.fill ins(%z : bf16) outs(%Ce : tensor<128x4096xbf16>) -> tensor<128x4096xbf16>
    %C = func.call @matmul(%A, %B, %C0)
        : (tensor<128x4096xbf16>, tensor<4096x4096xbf16>, tensor<128x4096xbf16>)
        -> tensor<128x4096xbf16>
    %v = tensor.extract %C[%c0, %c0] : tensor<128x4096xbf16>
    %f = arith.extf %v : bf16 to f32
    %i = arith.fptosi %f : f32 to i32
    return %i : i32
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %tiled, %ii, %jj, %kk = transform.structured.tile_using_for %mm
        tile_sizes [32, 64, 64]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                  !transform.any_op, !transform.any_op)
    transform.yield
  }
}

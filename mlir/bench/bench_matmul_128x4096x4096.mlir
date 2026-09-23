// matmul.mlir style at PACE production size: naive linalg.matmul, no tiles.
// C[128,4096] += A[128,4096] @ B[4096,4096]   bf16   FLOPs = 2*128*4096*4096
// Teaching matmul.mlir stays 2x2. This file is bench-only (alloc+fill, not dense constants).
// Element type stays bf16: linalg.matmul bf16 parses and lowers on AMD LLVM 20.

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

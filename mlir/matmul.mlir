// PACE TPP does:  Y = X @ W.T     (Linear, no bias)
// This MLIR does the same math as standard GEMM:
//   C = A @ B
// Match PACE by setting A = X, B = W.T
//
// Tiny 2x2 so we can JIT with mlir-runner (like mlir-learn/05_run.mlir).
//   A = [[1,2],[3,4]]  B = [[5,6],[7,8]]
//   C[0,0] = 1*5 + 2*7 = 19   <- @main returns this as i32
//
// This is NOT TPPLinear. It sits BESIDE PACE: same formula, MLIR compiler.

func.func @matmul(%A: tensor<2x2xf32>, %B: tensor<2x2xf32>,
                  %C: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<2x2xf32>, tensor<2x2xf32>)
                     outs(%C : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

func.func @main() -> i32 {
  %c0 = arith.constant 0 : index
  %A = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %B = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
  %zero = arith.constant dense<0.0> : tensor<2x2xf32>
  %C = func.call @matmul(%A, %B, %zero)
      : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %v = tensor.extract %C[%c0, %c0] : tensor<2x2xf32>
  %i = arith.fptosi %v : f32 to i32
  return %i : i32
}

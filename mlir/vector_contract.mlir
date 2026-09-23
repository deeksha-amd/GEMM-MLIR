// ISA hook: already at vector dialect.
// New instructions attach HERE (convert-vector-to-llvm=enable-x86vector
// or enable-amx), not by rewriting linalg.matmul.
//
// C[m,n] += A[m,k] * B[k,n]   kind = add  (the inner FMA of GEMM)

#mapA = affine_map<(m, n, k) -> (m, k)>
#mapB = affine_map<(m, n, k) -> (k, n)>
#mapC = affine_map<(m, n, k) -> (m, n)>

func.func @contract(%A: vector<8x8xf32>, %B: vector<8x8xf32>,
                    %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract
    {indexing_maps = [#mapA, #mapB, #mapC],
     iterator_types = ["parallel", "parallel", "reduction"],
     kind = #vector.kind<add>}
    %A, %B, %C : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

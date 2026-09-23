// One AVX-512 BF16 dot: the same mnemonic PACE/libXSMM JITs (vdpbf16ps).
// This is NOT a Linear/TPP kernel: no packing, tiling, BRGEMM, or @main.
// Does not replace matmul.mlir or vector_contract.mlir — LLVM dialect only.
//
//   dst[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]   (bf16 pairs -> f32, 16 lanes)
//
// Lower:  mlir-opt dpbf16ps.mlir | mlir-translate --mlir-to-llvmir
//         | llc -O3 -mcpu=x86-64-v4 -mattr=+avx512f,+avx512bf16

llvm.func @dot(%src: vector<16xf32>, %a: vector<32xbf16>, %b: vector<32xbf16>)
    -> vector<16xf32> {
  %r = llvm.call_intrinsic "llvm.x86.avx512bf16.dpbf16ps.512"(%src, %a, %b)
      : (vector<16xf32>, vector<32xbf16>, vector<32xbf16>) -> vector<16xf32>
  llvm.return %r : vector<16xf32>
}

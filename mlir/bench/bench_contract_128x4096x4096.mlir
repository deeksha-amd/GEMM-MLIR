// vector.contract style at production size.
// One vector.contract CANNOT be 128x4096 — that is not a register tile.
// Like-to-like: same 8x8 contract as the teaching file, scf tiles cover 128x4096x4096 bf16.
// Teaching vector_contract.mlir stays 8x8 (f32). No vdpbf16ps in this file.

#mapA = affine_map<(m, n, k) -> (m, k)>
#mapB = affine_map<(m, n, k) -> (k, n)>
#mapC = affine_map<(m, n, k) -> (m, n)>

func.func @contract(%A: vector<8x8xbf16>, %B: vector<8x8xbf16>,
                    %C: vector<8x8xbf16>) -> vector<8x8xbf16> {
  %0 = vector.contract
    {indexing_maps = [#mapA, #mapB, #mapC],
     iterator_types = ["parallel", "parallel", "reduction"],
     kind = #vector.kind<add>}
    %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xbf16>
  return %0 : vector<8x8xbf16>
}

func.func @main() -> i32 {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c4096 = arith.constant 4096 : index
  %one = arith.constant 1.0 : bf16
  %z = arith.constant 0.0 : bf16
  %A = memref.alloc() : memref<128x4096xbf16>
  %B = memref.alloc() : memref<4096x4096xbf16>
  %C = memref.alloc() : memref<128x4096xbf16>
  linalg.fill ins(%one : bf16) outs(%A : memref<128x4096xbf16>)
  linalg.fill ins(%one : bf16) outs(%B : memref<4096x4096xbf16>)
  linalg.fill ins(%z : bf16) outs(%C : memref<128x4096xbf16>)

  scf.for %i = %c0 to %c128 step %c8 {
    scf.for %j = %c0 to %c4096 step %c8 {
      %cvec0 = vector.transfer_read %C[%i, %j], %z {in_bounds = [true, true]}
          : memref<128x4096xbf16>, vector<8x8xbf16>
      %cvec = scf.for %k = %c0 to %c4096 step %c8
          iter_args(%acc = %cvec0) -> (vector<8x8xbf16>) {
        %a = vector.transfer_read %A[%i, %k], %z {in_bounds = [true, true]}
            : memref<128x4096xbf16>, vector<8x8xbf16>
        %b = vector.transfer_read %B[%k, %j], %z {in_bounds = [true, true]}
            : memref<4096x4096xbf16>, vector<8x8xbf16>
        %t = func.call @contract(%a, %b, %acc)
            : (vector<8x8xbf16>, vector<8x8xbf16>, vector<8x8xbf16>) -> vector<8x8xbf16>
        scf.yield %t : vector<8x8xbf16>
      }
      vector.transfer_write %cvec, %C[%i, %j] {in_bounds = [true, true]}
          : vector<8x8xbf16>, memref<128x4096xbf16>
    }
  }
  %v = memref.load %C[%c0, %c0] : memref<128x4096xbf16>
  %f = arith.extf %v : bf16 to f32
  %i32 = arith.fptosi %f : f32 to i32
  memref.dealloc %A : memref<128x4096xbf16>
  memref.dealloc %B : memref<4096x4096xbf16>
  memref.dealloc %C : memref<128x4096xbf16>
  return %i32 : i32
}

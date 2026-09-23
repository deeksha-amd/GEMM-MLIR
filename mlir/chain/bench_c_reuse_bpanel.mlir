// Same 1x16 vdpbf16ps tile, C in a zmm across K, B stored as N-panels so
// the K loop reads consecutive 64-byte lines (not stride-16KB rows).
module {
  func.func private @rtclock() -> f64
  func.func private @printF64(f64)
  func.func private @printNewline()

  func.func @matmul(%A: memref<128x2048x2xbf16>,
                    %Bp: memref<256x2048x32xbf16>,
                    %C: memref<128x4096xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    scf.for %nb = %c0 to %c256 step %c1 {
      %n = arith.muli %nb, %c16 : index
      scf.for %m = %c0 to %c128 step %c1 {
        %acc0 = vector.load %C[%m, %n] : memref<128x4096xf32>, vector<16xf32>
        %accN = scf.for %k = %c0 to %c2048 step %c1
            iter_args(%acc = %acc0) -> (vector<16xf32>) {
          %a2 = vector.load %A[%m, %k, %c0]
              : memref<128x2048x2xbf16>, vector<2xbf16>
          %ai = vector.bitcast %a2 : vector<2xbf16> to vector<1xi32>
          %ab = vector.broadcast %ai : vector<1xi32> to vector<16xi32>
          %av = vector.bitcast %ab : vector<16xi32> to vector<32xbf16>
          %bv = vector.load %Bp[%nb, %k, %c0]
              : memref<256x2048x32xbf16>, vector<32xbf16>
          %acc1 = x86vector.avx512.dot %acc, %av, %bv
              : vector<32xbf16> -> vector<16xf32>
          scf.yield %acc1 : vector<16xf32>
        }
        vector.store %accN, %C[%m, %n] : memref<128x4096xf32>, vector<16xf32>
      }
    }
    return
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %reps = arith.constant 1 : index
    %one = arith.constant 1.0 : bf16
    %zf = arith.constant 0.0 : f32
    %A = memref.alloc() {alignment = 64} : memref<128x2048x2xbf16>
    %Bp = memref.alloc() {alignment = 64} : memref<256x2048x32xbf16>
    %C = memref.alloc() {alignment = 64} : memref<128x4096xf32>
    linalg.fill ins(%one : bf16) outs(%A : memref<128x2048x2xbf16>)
    linalg.fill ins(%one : bf16) outs(%Bp : memref<256x2048x32xbf16>)
    linalg.fill ins(%zf : f32) outs(%C : memref<128x4096xf32>)
    func.call @matmul(%A, %Bp, %C)
        : (memref<128x2048x2xbf16>, memref<256x2048x32xbf16>, memref<128x4096xf32>) -> ()
    %t0 = func.call @rtclock() : () -> f64
    scf.for %r = %c0 to %reps step %c1 {
      func.call @matmul(%A, %Bp, %C)
          : (memref<128x2048x2xbf16>, memref<256x2048x32xbf16>, memref<128x4096xf32>) -> ()
    }
    %t1 = func.call @rtclock() : () -> f64
    %dt = arith.subf %t1, %t0 : f64
    func.call @printF64(%dt) : (f64) -> ()
    func.call @printNewline() : () -> ()
    %v = memref.load %C[%c0, %c0] : memref<128x4096xf32>
    %vf = arith.extf %v : f32 to f64
    func.call @printF64(%vf) : (f64) -> ()
    func.call @printNewline() : () -> ()
    memref.dealloc %A : memref<128x2048x2xbf16>
    memref.dealloc %Bp : memref<256x2048x32xbf16>
    memref.dealloc %C : memref<128x4096xf32>
    return
  }
}

// C reused in a zmm across K: vector<16xf32> iter_arg, vector.load/store only
// outside the K loop. Same 1x16 tile / vdpbf16ps as the automatic chain.
module {
  func.func private @rtclock() -> f64
  func.func private @printF64(f64)
  func.func private @printNewline()

  func.func @matmul(%A: memref<128x2048x2xbf16>,
                    %B: memref<2048x4096x2xbf16>,
                    %C: memref<128x4096xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c2048 = arith.constant 2048 : index
    %c4096 = arith.constant 4096 : index
    %Bflat = memref.collapse_shape %B [[0], [1, 2]]
        : memref<2048x4096x2xbf16> into memref<2048x8192xbf16>
    scf.for %n = %c0 to %c4096 step %c16 {
      scf.for %m = %c0 to %c128 step %c1 {
        %acc0 = vector.load %C[%m, %n] : memref<128x4096xf32>, vector<16xf32>
        %accN = scf.for %k = %c0 to %c2048 step %c1
            iter_args(%acc = %acc0) -> (vector<16xf32>) {
          %a2 = vector.load %A[%m, %k, %c0]
              : memref<128x2048x2xbf16>, vector<2xbf16>
          %ai = vector.bitcast %a2 : vector<2xbf16> to vector<1xi32>
          %ab = vector.broadcast %ai : vector<1xi32> to vector<16xi32>
          %av = vector.bitcast %ab : vector<16xi32> to vector<32xbf16>
          %n2 = arith.muli %n, %c2 : index
          %bv = vector.load %Bflat[%k, %n2]
              : memref<2048x8192xbf16>, vector<32xbf16>
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
    %B = memref.alloc() {alignment = 64} : memref<2048x4096x2xbf16>
    %C = memref.alloc() {alignment = 64} : memref<128x4096xf32>
    linalg.fill ins(%one : bf16) outs(%A : memref<128x2048x2xbf16>)
    linalg.fill ins(%one : bf16) outs(%B : memref<2048x4096x2xbf16>)
    linalg.fill ins(%zf : f32) outs(%C : memref<128x4096xf32>)
    func.call @matmul(%A, %B, %C) : (memref<128x2048x2xbf16>, memref<2048x4096x2xbf16>, memref<128x4096xf32>) -> ()
    %t0 = func.call @rtclock() : () -> f64
    scf.for %r = %c0 to %reps step %c1 {
      func.call @matmul(%A, %B, %C) : (memref<128x2048x2xbf16>, memref<2048x4096x2xbf16>, memref<128x4096xf32>) -> ()
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
    memref.dealloc %B : memref<2048x4096x2xbf16>
    memref.dealloc %C : memref<128x4096xf32>
    return
  }
}

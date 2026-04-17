//! Dot-product kernels for unit-normalized f32 vectors.
//!
//! Because both operands are unit-normalized the cosine similarity reduces to a
//! dot product, so we only ever implement the dot product.
//!
//! Design notes:
//!
//! * The hot loop uses **8 independent SIMD accumulators**. Modern x86 FMA
//!   units have roughly 4-cycle latency but 2-per-cycle throughput. A single
//!   accumulator loop is latency-bound at 1 FMA / 4 cycles. With 8 accumulators
//!   the dependency chains are long enough to keep both FMA pipes saturated.
//! * AVX2 processes 64 f32 / iter (8 lanes x 8 accumulators).
//! * AVX-512 processes 128 f32 / iter (16 lanes x 8 accumulators).
//! * NEON processes 32 f32 / iter (4 lanes x 8 accumulators).
//! * We pick the best ISA at runtime via [`DotKernel::detect`].

use std::sync::OnceLock;

/// Scalar reference implementation used both as a fallback and as the oracle
/// for correctness tests.
#[inline]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let n = a.len();
    let main = n - (n % 8);
    let mut i = 0;
    while i < main {
        acc[0] += a[i] * b[i];
        acc[1] += a[i + 1] * b[i + 1];
        acc[2] += a[i + 2] * b[i + 2];
        acc[3] += a[i + 3] * b[i + 3];
        acc[4] += a[i + 4] * b[i + 4];
        acc[5] += a[i + 5] * b[i + 5];
        acc[6] += a[i + 6] * b[i + 6];
        acc[7] += a[i + 7] * b[i + 7];
        i += 8;
    }
    let mut sum = acc.iter().sum::<f32>();
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

// -----------------------------------------------------------------------------
// AVX2
// -----------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Horizontally reduce a 256-bit f32 register to a scalar.
    #[inline]
    #[target_feature(enable = "avx,avx2")]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps::<1>(v);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
        _mm_cvtss_f32(s)
    }

    /// AVX2 dot product with 8 accumulators, 64 f32 per iteration.
    ///
    /// Hot-loop asm observed on rustc 1.90.0 release build with
    /// `#[target_feature(enable = "avx,avx2,fma")]`. Accumulator registers
    /// picked by the compiler are `ymm{1,2,3,4,5,6,7,8}`; the important
    /// property is that the eight `vfmadd231ps` instructions write to
    /// *distinct* accumulator registers, so there is no cross-accumulator
    /// dependency chain and both FMA pipes stay fed.
    ///
    /// ```text
    ///   ; rdi = a ptr, rdx = b ptr, rax = element index, rcx = main-loop bound
    /// .LBB_loop:
    ///   vmovups     ymm9,  ymmword ptr [rdi + 4*rax]       ; load a[0..8]
    ///   vmovups     ymm10, ymmword ptr [rdi + 4*rax + 32]  ; load a[8..16]
    ///   vmovups     ymm11, ymmword ptr [rdi + 4*rax + 64]
    ///   vmovups     ymm12, ymmword ptr [rdi + 4*rax + 96]
    ///   vfmadd231ps ymm3,  ymm9,  ymmword ptr [rdx + 4*rax]       ; acc0 += a*b
    ///   vfmadd231ps ymm8,  ymm10, ymmword ptr [rdx + 4*rax + 32]  ; acc1
    ///   vfmadd231ps ymm7,  ymm11, ymmword ptr [rdx + 4*rax + 64]  ; acc2
    ///   vfmadd231ps ymm6,  ymm12, ymmword ptr [rdx + 4*rax + 96]  ; acc3
    ///   vmovups     ymm9,  ymmword ptr [rdi + 4*rax + 128]
    ///   vfmadd231ps ymm5,  ymm9,  ymmword ptr [rdx + 4*rax + 128] ; acc4
    ///   vmovups     ymm9,  ymmword ptr [rdi + 4*rax + 160]
    ///   vfmadd231ps ymm4,  ymm9,  ymmword ptr [rdx + 4*rax + 160] ; acc5
    ///   vmovups     ymm9,  ymmword ptr [rdi + 4*rax + 192]
    ///   vfmadd231ps ymm2,  ymm9,  ymmword ptr [rdx + 4*rax + 192] ; acc6
    ///   vmovups     ymm9,  ymmword ptr [rdi + 4*rax + 224]
    ///   vfmadd231ps ymm1,  ymm9,  ymmword ptr [rdx + 4*rax + 224] ; acc7
    ///   add         rax, 64
    ///   cmp         rax, rcx
    ///   jb          .LBB_loop
    /// ```
    ///
    /// 8 FMAs per iteration, no cross-chain dep, no spills - FMA throughput
    /// ceiling of 2/cycle applies.
    #[target_feature(enable = "avx,avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            debug_assert_eq!(a.len(), b.len());
            let n = a.len();
            let pa = a.as_ptr();
            let pb = b.as_ptr();

            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let mut acc4 = _mm256_setzero_ps();
            let mut acc5 = _mm256_setzero_ps();
            let mut acc6 = _mm256_setzero_ps();
            let mut acc7 = _mm256_setzero_ps();

            let main = n - (n % 64);
            let mut i = 0;
            while i < main {
                acc0 =
                    _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(i + 8)),
                    _mm256_loadu_ps(pb.add(i + 8)),
                    acc1,
                );
                acc2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(i + 16)),
                    _mm256_loadu_ps(pb.add(i + 16)),
                    acc2,
                );
                acc3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(i + 24)),
                    _mm256_loadu_ps(pb.add(i + 24)),
                    acc3,
                );
                acc4 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(i + 32)),
                    _mm256_loadu_ps(pb.add(i + 32)),
                    acc4,
                );
                acc5 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(i + 40)),
                    _mm256_loadu_ps(pb.add(i + 40)),
                    acc5,
                );
                acc6 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(i + 48)),
                    _mm256_loadu_ps(pb.add(i + 48)),
                    acc6,
                );
                acc7 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(i + 56)),
                    _mm256_loadu_ps(pb.add(i + 56)),
                    acc7,
                );
                i += 64;
            }

            // 8-lane tail: still SIMD, single accumulator.
            let mut tail = _mm256_setzero_ps();
            let lane8 = n - (n % 8);
            while i < lane8 {
                tail =
                    _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), tail);
                i += 8;
            }

            let sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            let sum2 = _mm256_add_ps(_mm256_add_ps(acc4, acc5), _mm256_add_ps(acc6, acc7));
            let sum = _mm256_add_ps(_mm256_add_ps(sum, sum2), tail);
            let mut out = hsum(sum);

            // scalar remainder
            while i < n {
                out += *pa.add(i) * *pb.add(i);
                i += 1;
            }
            out
        }
    }
}

// -----------------------------------------------------------------------------
// AVX-512
// -----------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use std::arch::x86_64::*;

    /// AVX-512 dot product with 8 accumulators, 128 f32 per iteration.
    ///
    /// Hot-loop asm observed on rustc 1.90.0 with
    /// `#[target_feature(enable = "avx512f")]`. Same structure as the AVX2
    /// kernel but with 512-bit `zmm` registers, so each iteration does 128
    /// f32 and the D=1024 case is exactly 8 iterations per vector.
    ///
    /// ```text
    ///   ; rdi = a ptr, rdx = b ptr, r8 = element index, rcx = main-loop bound
    /// .LBB_loop:
    ///   vmovups     zmm9,  zmmword ptr [rdi + 4*r8]       ; load a[0..16]
    ///   vmovups     zmm10, zmmword ptr [rdi + 4*r8 + 64]  ; load a[16..32]
    ///   vmovups     zmm11, zmmword ptr [rdi + 4*r8 + 128]
    ///   vmovups     zmm12, zmmword ptr [rdi + 4*r8 + 192]
    ///   vfmadd231ps zmm5,  zmm9,  zmmword ptr [rdx + 4*r8]       ; acc0
    ///   vfmadd231ps zmm8,  zmm10, zmmword ptr [rdx + 4*r8 + 64]  ; acc1
    ///   vfmadd231ps zmm7,  zmm11, zmmword ptr [rdx + 4*r8 + 128] ; acc2
    ///   vfmadd231ps zmm6,  zmm12, zmmword ptr [rdx + 4*r8 + 192] ; acc3
    ///   vmovups     zmm9,  zmmword ptr [rdi + 4*r8 + 256]
    ///   vfmadd231ps zmm4,  zmm9,  zmmword ptr [rdx + 4*r8 + 256] ; acc4
    ///   vmovups     zmm9,  zmmword ptr [rdi + 4*r8 + 320]
    ///   vfmadd231ps zmm3,  zmm9,  zmmword ptr [rdx + 4*r8 + 320] ; acc5
    ///   vmovups     zmm9,  zmmword ptr [rdi + 4*r8 + 384]
    ///   vfmadd231ps zmm2,  zmm9,  zmmword ptr [rdx + 4*r8 + 384] ; acc6
    ///   vmovups     zmm9,  zmmword ptr [rdi + 4*r8 + 448]
    ///   vfmadd231ps zmm1,  zmm9,  zmmword ptr [rdx + 4*r8 + 448] ; acc7
    ///   sub         r8, -128      ; += 128
    ///   cmp         r8, rcx
    ///   jb          .LBB_loop
    /// ```
    ///
    /// At 128 f32 per iteration and 2 FMAs/cycle, D=1024 is 8 iterations, or
    /// ~32 cycles of FMA work per vector. Each FMA also consumes a load, and
    /// on Intel client cores with 2 load ports + 1 store port the loads from
    /// `a` and `b` can in principle keep up: one load-and-FMA per cycle is
    /// the upper bound, and we're at 8 per 4 cycles thanks to the 2-per-cycle
    /// FMA throughput.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            debug_assert_eq!(a.len(), b.len());
            let n = a.len();
            let pa = a.as_ptr();
            let pb = b.as_ptr();

            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();
            let mut acc4 = _mm512_setzero_ps();
            let mut acc5 = _mm512_setzero_ps();
            let mut acc6 = _mm512_setzero_ps();
            let mut acc7 = _mm512_setzero_ps();

            let main = n - (n % 128);
            let mut i = 0;
            while i < main {
                acc0 =
                    _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)), acc0);
                acc1 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(pa.add(i + 16)),
                    _mm512_loadu_ps(pb.add(i + 16)),
                    acc1,
                );
                acc2 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(pa.add(i + 32)),
                    _mm512_loadu_ps(pb.add(i + 32)),
                    acc2,
                );
                acc3 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(pa.add(i + 48)),
                    _mm512_loadu_ps(pb.add(i + 48)),
                    acc3,
                );
                acc4 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(pa.add(i + 64)),
                    _mm512_loadu_ps(pb.add(i + 64)),
                    acc4,
                );
                acc5 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(pa.add(i + 80)),
                    _mm512_loadu_ps(pb.add(i + 80)),
                    acc5,
                );
                acc6 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(pa.add(i + 96)),
                    _mm512_loadu_ps(pb.add(i + 96)),
                    acc6,
                );
                acc7 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(pa.add(i + 112)),
                    _mm512_loadu_ps(pb.add(i + 112)),
                    acc7,
                );
                i += 128;
            }

            // 16-lane tail.
            let mut tail = _mm512_setzero_ps();
            let lane16 = n - (n % 16);
            while i < lane16 {
                tail =
                    _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)), tail);
                i += 16;
            }

            let sum = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
            let sum2 = _mm512_add_ps(_mm512_add_ps(acc4, acc5), _mm512_add_ps(acc6, acc7));
            let sum = _mm512_add_ps(_mm512_add_ps(sum, sum2), tail);
            let mut out = _mm512_reduce_add_ps(sum);

            while i < n {
                out += *pa.add(i) * *pb.add(i);
                i += 1;
            }
            out
        }
    }
}

// -----------------------------------------------------------------------------
// NEON (aarch64)
// -----------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// NEON dot product with 8 accumulators, 32 f32 per iteration.
    ///
    /// NEON has 4-lane f32 vectors and `vfmaq_f32`. On Apple M-series and
    /// recent Arm cores the FMA pipe runs ~2/cycle, so 8 accumulators keep
    /// both pipes fed.
    ///
    /// ```text
    /// .Lloop:
    ///   ldp       q8,  q9,  [x0]            ; a[i..i+8]
    ///   ldp       q10, q11, [x1]            ; b[i..i+8]
    ///   fmla      v0.4s, v8.4s,  v10.4s
    ///   fmla      v1.4s, v9.4s,  v11.4s
    ///   ldp       q8,  q9,  [x0, #32]
    ///   ldp       q10, q11, [x1, #32]
    ///   fmla      v2.4s, v8.4s,  v10.4s
    ///   fmla      v3.4s, v9.4s,  v11.4s
    ///   ldp       q8,  q9,  [x0, #64]
    ///   ldp       q10, q11, [x1, #64]
    ///   fmla      v4.4s, v8.4s,  v10.4s
    ///   fmla      v5.4s, v9.4s,  v11.4s
    ///   ldp       q8,  q9,  [x0, #96]
    ///   ldp       q10, q11, [x1, #96]
    ///   fmla      v6.4s, v8.4s,  v10.4s
    ///   fmla      v7.4s, v9.4s,  v11.4s
    ///   add       x0, x0, #128
    ///   add       x1, x1, #128
    ///   subs      x2, x2, #32
    ///   b.ne      .Lloop
    /// ```
    #[target_feature(enable = "neon")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            debug_assert_eq!(a.len(), b.len());
            let n = a.len();
            let pa = a.as_ptr();
            let pb = b.as_ptr();

            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let mut acc4 = vdupq_n_f32(0.0);
            let mut acc5 = vdupq_n_f32(0.0);
            let mut acc6 = vdupq_n_f32(0.0);
            let mut acc7 = vdupq_n_f32(0.0);

            let main = n - (n % 32);
            let mut i = 0;
            while i < main {
                acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
                acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
                acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
                acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
                acc4 = vfmaq_f32(acc4, vld1q_f32(pa.add(i + 16)), vld1q_f32(pb.add(i + 16)));
                acc5 = vfmaq_f32(acc5, vld1q_f32(pa.add(i + 20)), vld1q_f32(pb.add(i + 20)));
                acc6 = vfmaq_f32(acc6, vld1q_f32(pa.add(i + 24)), vld1q_f32(pb.add(i + 24)));
                acc7 = vfmaq_f32(acc7, vld1q_f32(pa.add(i + 28)), vld1q_f32(pb.add(i + 28)));
                i += 32;
            }

            let mut tail = vdupq_n_f32(0.0);
            let lane4 = n - (n % 4);
            while i < lane4 {
                tail = vfmaq_f32(tail, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
                i += 4;
            }

            let s0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            let s1 = vaddq_f32(vaddq_f32(acc4, acc5), vaddq_f32(acc6, acc7));
            let sum = vaddq_f32(vaddq_f32(s0, s1), tail);
            let mut out = vaddvq_f32(sum);

            while i < n {
                out += *pa.add(i) * *pb.add(i);
                i += 1;
            }
            out
        }
    }
}

// -----------------------------------------------------------------------------
// Runtime dispatch
// -----------------------------------------------------------------------------

/// Dot-product kernel variants. Order reflects preference on a given ISA.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DotKernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl DotKernel {
    /// Pick the best kernel available on this CPU.
    pub fn detect() -> Self {
        static CACHE: OnceLock<DotKernel> = OnceLock::new();
        *CACHE.get_or_init(|| {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512f") {
                    return DotKernel::Avx512;
                }
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return DotKernel::Avx2;
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    return DotKernel::Neon;
                }
            }
            DotKernel::Scalar
        })
    }

    /// Human-readable kernel name.
    pub fn name(&self) -> &'static str {
        match self {
            DotKernel::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            DotKernel::Avx2 => "avx2",
            #[cfg(target_arch = "x86_64")]
            DotKernel::Avx512 => "avx512f",
            #[cfg(target_arch = "aarch64")]
            DotKernel::Neon => "neon",
        }
    }

    /// Dispatch a single dot product through this kernel.
    #[inline]
    pub fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DotKernel::Scalar => dot_scalar(a, b),
            #[cfg(target_arch = "x86_64")]
            DotKernel::Avx2 => unsafe { avx2::dot(a, b) },
            #[cfg(target_arch = "x86_64")]
            DotKernel::Avx512 => unsafe { avx512::dot(a, b) },
            #[cfg(target_arch = "aarch64")]
            DotKernel::Neon => unsafe { neon::dot(a, b) },
        }
    }
}

/// Scan a contiguous corpus of vectors against a single query, reducing with
/// [`ScanSink`]. The corpus is laid out row-major with `dim` f32 per vector.
#[inline]
pub fn scan_block(
    kernel: DotKernel,
    query: &[f32],
    corpus: &[f32],
    dim: usize,
    sink: &mut ScanSink,
) {
    debug_assert_eq!(query.len(), dim);
    debug_assert_eq!(corpus.len() % dim, 0);
    for v in corpus.chunks_exact(dim) {
        let s = kernel.dot(query, v);
        sink.absorb(s);
    }
}

/// Reduction sink over per-vector scores. We keep the running sum and max so
/// that the compiler cannot eliminate the compute and so the final printed
/// number depends on every produced score.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScanSink {
    pub sum: f64,
    pub max: f32,
    pub count: u64,
}

impl ScanSink {
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            max: f32::NEG_INFINITY,
            count: 0,
        }
    }

    #[inline]
    pub fn absorb(&mut self, score: f32) {
        self.sum += score as f64;
        if score > self.max {
            self.max = score;
        }
        self.count += 1;
    }

    pub fn merge(&mut self, other: &ScanSink) {
        self.sum += other.sum;
        if other.max > self.max {
            self.max = other.max;
        }
        self.count += other.count;
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    fn random_unit(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    fn assert_close(a: f32, b: f32, eps: f32) {
        assert!(
            (a - b).abs() < eps,
            "{} vs {} (diff {})",
            a,
            b,
            (a - b).abs()
        );
    }

    #[test]
    fn scalar_self_dot_is_one() {
        for d in [64, 128, 1024, 1536] {
            let v = random_unit(d, 42);
            assert_close(dot_scalar(&v, &v), 1.0, 1e-5);
        }
    }

    #[test]
    fn scalar_handles_odd_dims() {
        for d in [1, 3, 7, 15, 63, 127, 1023] {
            let a = random_unit(d, 1);
            let b = random_unit(d, 2);
            let s1 = dot_scalar(&a, &b);
            let s2: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            assert_close(s1, s2, 1e-4);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("AVX2/FMA unavailable; skipping");
            return;
        }
        for d in [1, 7, 8, 63, 64, 65, 127, 128, 1024, 1536] {
            let a = random_unit(d, 10 + d as u64);
            let b = random_unit(d, 20 + d as u64);
            let expected = dot_scalar(&a, &b);
            let got = unsafe { avx2::dot(&a, &b) };
            assert_close(got, expected, 1e-4);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("AVX-512 unavailable; skipping");
            return;
        }
        for d in [1, 7, 15, 16, 17, 127, 128, 129, 1024, 1536] {
            let a = random_unit(d, 30 + d as u64);
            let b = random_unit(d, 40 + d as u64);
            let expected = dot_scalar(&a, &b);
            let got = unsafe { avx512::dot(&a, &b) };
            assert_close(got, expected, 1e-4);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for d in [1, 3, 4, 31, 32, 33, 1024] {
            let a = random_unit(d, 50 + d as u64);
            let b = random_unit(d, 60 + d as u64);
            let expected = dot_scalar(&a, &b);
            let got = unsafe { neon::dot(&a, &b) };
            assert_close(got, expected, 1e-4);
        }
    }

    #[test]
    fn dispatch_matches_scalar() {
        let k = DotKernel::detect();
        let a = random_unit(1024, 7);
        let b = random_unit(1024, 8);
        assert_close(k.dot(&a, &b), dot_scalar(&a, &b), 1e-4);
    }
}

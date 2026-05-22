// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for "mask invalid values to zero, then cast" kernels.
//!
//! The operation: given `values: &[F]` and a validity bitmask, produce a
//! `Buffer<T>` where valid positions hold `values[i] as T` and invalid
//! positions hold `0`.
//!
//! The interesting question is whether the kernel auto-vectorizes. Zipping a
//! value slice with `BitBuffer::iter()` (one bool per bit, produced by
//! shift+extract) is a vectorization blocker. Bulk-expanding the bitmap into a
//! contiguous factor slice first lets the subsequent multiply/cast loop turn
//! into SIMD. The benches below measure that gap.

use std::ops::BitAnd;

use divan::Bencher;
use num_traits::AsPrimitive;
use num_traits::NumCast;
use rand::prelude::*;
use vortex_array::dtype::NativePType;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

/// Types whose "zero out if invalid" is a bitwise AND with a per-element
/// all-ones / all-zero mask word. Ints AND directly; floats go through their
/// bit pattern. This is the `Bits` associated type that would have to be added
/// to `NativePType` to write the masking generically; kept local to the
/// benchmark to avoid growing public API for an experiment.
trait Maskable: NativePType {
    /// Unsigned integer with the same width as `Self`.
    type Bits: NativePType + BitAnd<Output = Self::Bits>;
    fn to_bits(self) -> Self::Bits;
    fn from_bits(bits: Self::Bits) -> Self;
    /// All bits clear when `keep == 0`, all bits set when `keep == 1`.
    fn mask_word(keep: u8) -> Self::Bits;
}

macro_rules! impl_maskable_int {
    ($ty:ty, $bits:ty) => {
        impl Maskable for $ty {
            type Bits = $bits;
            #[inline(always)]
            fn to_bits(self) -> $bits {
                self as $bits
            }
            #[inline(always)]
            fn from_bits(bits: $bits) -> Self {
                bits as $ty
            }
            #[inline(always)]
            fn mask_word(keep: u8) -> $bits {
                (keep as $bits).wrapping_neg()
            }
        }
    };
}

impl_maskable_int!(i32, u32);
impl_maskable_int!(i64, u64);

impl Maskable for f32 {
    type Bits = u32;
    #[inline(always)]
    fn to_bits(self) -> u32 {
        f32::to_bits(self)
    }
    #[inline(always)]
    fn from_bits(bits: u32) -> Self {
        f32::from_bits(bits)
    }
    #[inline(always)]
    fn mask_word(keep: u8) -> u32 {
        (keep as u32).wrapping_neg()
    }
}

fn main() {
    divan::main();
}

const N: usize = 100_000;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Floor: pure cast, ignores validity. Lower bound on achievable time.
/// This is the loop that today's `cast.rs` uses after an up-front
/// `values_fit_in` range check.
#[inline(never)]
fn k_as_only<F, T>(values: &[F]) -> Buffer<T>
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    BufferMut::from_trusted_len_iter(values.iter().map(|&v| v.as_())).freeze()
}

/// Fused "check + cast" fast path: cast every element and, in the same pass,
/// detect whether any conversion was lossy via a branchless round-trip compare
/// `(v as T) as F == v`. No bitmask is consumed and no min/max pass is needed;
/// the validity-aware path is only required when `lossy` is set *and* the
/// offending element is at a valid position (the rare error case).
#[inline(never)]
fn k_check_cast_fused<F, T>(values: &[F]) -> (Buffer<T>, bool)
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType + AsPrimitive<F>,
{
    let mut lossy = false;
    let out = BufferMut::from_trusted_len_iter(values.iter().map(|&v| {
        let t: T = v.as_();
        let back: F = t.as_();
        lossy |= !back.is_eq(v);
        t
    }))
    .freeze();
    (out, lossy)
}

/// The snippet under discussion: zip values with `BitBuffer::iter()`, multiply
/// by a 0/1 factor, then a *checked* `NumCast::from`, falling back to `as_()`.
/// The bit iterator and the checked cast both block vectorization.
#[inline(never)]
fn k_zip_numcast<F, T>(values: &[F], mask: &BitBuffer) -> Buffer<T>
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    BufferMut::from_trusted_len_iter(values.iter().zip(mask.iter()).map(|(&v, valid)| {
        let factor = if valid { F::one() } else { F::zero() };
        <T as NumCast>::from(v * factor).unwrap_or_else(|| v.as_())
    }))
    .freeze()
}

/// Same zip-with-bit-iterator structure, but a branch/select instead of the
/// multiply + checked cast. Isolates the cost of the bit iterator itself.
#[inline(never)]
fn k_zip_select<F, T>(values: &[F], mask: &BitBuffer) -> Buffer<T>
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    BufferMut::from_trusted_len_iter(values.iter().zip(mask.iter()).map(|(&v, valid)| {
        if valid { v.as_() } else { T::zero() }
    }))
    .freeze()
}

/// Bulk-expand the packed bitmap into a contiguous `Buffer<F>` of 0/1 factors,
/// then do a tight slice-vs-slice zip. The second loop is a pure
/// `[F] x [F] -> [T]` map and should auto-vectorize.
///
/// Assumes `mask.offset() == 0` (true for the data this bench constructs).
#[inline(never)]
fn k_bulk_mul<F, T>(values: &[F], mask: &BitBuffer) -> Buffer<T>
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    let factors = expand_factors::<F>(mask);
    BufferMut::from_trusted_len_iter(
        values
            .iter()
            .zip(factors.iter())
            .map(|(&v, &factor)| (v * factor).as_()),
    )
    .freeze()
}

/// Like `k_bulk_mul` but the second pass uses a branchless slice select rather
/// than a multiply, so it is correct for floats (no `NaN * 0` surprises).
#[inline(never)]
fn k_bulk_select<F, T>(values: &[F], mask: &BitBuffer) -> Buffer<T>
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    let keep = expand_keep(mask);
    BufferMut::from_trusted_len_iter(
        values
            .iter()
            .zip(keep.iter())
            .map(|(&v, &k)| if k != 0 { v.as_() } else { T::zero() }),
    )
    .freeze()
}

/// Expand a packed bitmap into a `Buffer<F>` of `F::one()` / `F::zero()`.
///
/// Writes into a pre-sized slice in 8-element chunks. The inner constant-trip
/// loop unrolls and the per-byte expansion vectorizes, unlike the per-bit
/// `BitBuffer::iter()` path.
fn expand_factors<F: NativePType>(mask: &BitBuffer) -> Buffer<F> {
    let len = mask.len();
    let bytes = mask.inner().as_slice();
    let mut out = BufferMut::<F>::zeroed(len);
    let dst = out.as_mut_slice();
    let one = F::one();
    let zero = F::zero();
    let full = len / 8;
    for (chunk, &b) in dst.chunks_exact_mut(8).zip(&bytes[..full]) {
        for (j, slot) in chunk.iter_mut().enumerate() {
            *slot = if (b >> j) & 1 == 1 { one } else { zero };
        }
    }
    for i in (full * 8)..len {
        dst[i] = if (bytes[i >> 3] >> (i & 7)) & 1 == 1 {
            one
        } else {
            zero
        };
    }
    out.freeze()
}

/// Expand a packed bitmap into a `Buffer<u8>` of 0/1 keep flags, 8 bytes at a
/// time so the expansion auto-vectorizes.
fn expand_keep(mask: &BitBuffer) -> Buffer<u8> {
    let len = mask.len();
    let bytes = mask.inner().as_slice();
    let mut out = BufferMut::<u8>::zeroed(len);
    let dst = out.as_mut_slice();
    let full = len / 8;
    for (chunk, &b) in dst.chunks_exact_mut(8).zip(&bytes[..full]) {
        for (j, slot) in chunk.iter_mut().enumerate() {
            *slot = (b >> j) & 1;
        }
    }
    for i in (full * 8)..len {
        dst[i] = (bytes[i >> 3] >> (i & 7)) & 1;
    }
    out.freeze()
}

/// Floor for the same-type masking kernels: copy values, apply no validity.
#[inline(never)]
fn k_copy<F: NativePType>(values: &[F]) -> Buffer<F> {
    BufferMut::from_trusted_len_iter(values.iter().copied()).freeze()
}

/// Two-pass bitwise-AND masking: expand the bitmap into a `u8` keep-mask
/// (vectorized), then `from_bits(to_bits(v) & mask_word(keep))`.
#[inline(never)]
fn k_bitand<F: Maskable>(values: &[F], mask: &BitBuffer) -> Buffer<F> {
    let keep = expand_keep(mask);
    BufferMut::from_trusted_len_iter(
        values
            .iter()
            .zip(keep.iter())
            .map(|(&v, &k)| F::from_bits(F::to_bits(v) & F::mask_word(k))),
    )
    .freeze()
}

/// Single-pass bitwise-AND masking: walk the packed bitmap a byte (8 elements)
/// at a time, expanding each bit to a full-width mask word inline. No second
/// buffer, one pass over the data.
#[inline(never)]
fn k_bitand_fused<F: Maskable>(values: &[F], mask: &BitBuffer) -> Buffer<F> {
    let len = values.len();
    let bytes = mask.inner().as_slice();
    let full = len / 8;
    let mut out = BufferMut::<F>::zeroed(len);
    let dst = out.as_mut_slice();
    for ((vb, ob), &mb) in values[..full * 8]
        .chunks_exact(8)
        .zip(dst.chunks_exact_mut(8))
        .zip(&bytes[..full])
    {
        for (j, slot) in ob.iter_mut().enumerate() {
            *slot = F::from_bits(F::to_bits(vb[j]) & F::mask_word((mb >> j) & 1));
        }
    }
    for i in (full * 8)..len {
        let k = (bytes[i >> 3] >> (i & 7)) & 1;
        dst[i] = F::from_bits(F::to_bits(values[i]) & F::mask_word(k));
    }
    out.freeze()
}

/// Expand the bitmap into full-width `0` / `!0` mask words, 8 at a time.
fn expand_words<F: Maskable>(mask: &BitBuffer) -> Buffer<F::Bits> {
    let len = mask.len();
    let bytes = mask.inner().as_slice();
    let mut out = BufferMut::<F::Bits>::zeroed(len);
    let dst = out.as_mut_slice();
    let full = len / 8;
    for (chunk, &b) in dst.chunks_exact_mut(8).zip(&bytes[..full]) {
        for (j, slot) in chunk.iter_mut().enumerate() {
            *slot = F::mask_word((b >> j) & 1);
        }
    }
    for i in (full * 8)..len {
        dst[i] = F::mask_word((bytes[i >> 3] >> (i & 7)) & 1);
    }
    out.freeze()
}

/// Two-pass, but the apply loop is a *pure* `vpand` slice zip (no `vpsubd`):
/// expand to `0` / `!0` mask words, then `from_bits(to_bits(v) & word)`.
#[inline(never)]
fn k_bitand_words<F: Maskable>(values: &[F], mask: &BitBuffer) -> Buffer<F> {
    let words = expand_words::<F>(mask);
    BufferMut::from_trusted_len_iter(
        values
            .iter()
            .zip(words.iter())
            .map(|(&v, &w)| F::from_bits(F::to_bits(v) & w)),
    )
    .freeze()
}

/// Single pass over main memory: process 64 elements per step with a small
/// on-stack `keep` temp, so both the bit-expand and the AND vectorize while the
/// large intermediate buffer disappears (2N memory traffic instead of 3N).
#[inline(never)]
fn k_bitand_blocked<F: Maskable>(values: &[F], mask: &BitBuffer) -> Buffer<F> {
    let len = values.len();
    let bytes = mask.inner().as_slice();
    let blocks = len / 64;
    let mut out = BufferMut::<F>::zeroed(len);
    let dst = out.as_mut_slice();
    for blk in 0..blocks {
        let mb = &bytes[blk * 8..blk * 8 + 8];
        let vb = &values[blk * 64..blk * 64 + 64];
        let ob = &mut dst[blk * 64..blk * 64 + 64];
        let mut keep = [F::Bits::default(); 64];
        for (i, k) in keep.iter_mut().enumerate() {
            *k = F::mask_word((mb[i >> 3] >> (i & 7)) & 1);
        }
        for ((o, &v), &k) in ob.iter_mut().zip(vb).zip(keep.iter()) {
            *o = F::from_bits(F::to_bits(v) & k);
        }
    }
    for i in (blocks * 64)..len {
        let k = F::mask_word((bytes[i >> 3] >> (i & 7)) & 1);
        dst[i] = F::from_bits(F::to_bits(values[i]) & k);
    }
    out.freeze()
}

// ---------------------------------------------------------------------------
// Data + bench registration
// ---------------------------------------------------------------------------

fn gen_values<F>(rng: &mut StdRng) -> Buffer<F>
where
    F: NativePType,
    u16: AsPrimitive<F>,
{
    BufferMut::from_trusted_len_iter((0..N).map(|_| {
        let v: u16 = rng.random();
        v.as_()
    }))
    .freeze()
}

fn gen_mask(rng: &mut StdRng) -> BitBuffer {
    BitBuffer::from((0..N).map(|_| rng.random_bool(0.5)).collect::<Vec<bool>>())
}

macro_rules! bench_pair {
    ($mod_name:ident, $F:ty, $T:ty) => {
        mod $mod_name {
            use super::*;

            fn inputs() -> (Buffer<$F>, BitBuffer) {
                let mut rng = StdRng::seed_from_u64(42);
                (gen_values::<$F>(&mut rng), gen_mask(&mut rng))
            }

            #[divan::bench]
            fn as_only(bencher: Bencher) {
                let (values, _) = inputs();
                bencher.bench(|| k_as_only::<$F, $T>(values.as_slice()));
            }

            #[divan::bench]
            fn check_cast_fused(bencher: Bencher) {
                let (values, _) = inputs();
                bencher.bench(|| k_check_cast_fused::<$F, $T>(values.as_slice()));
            }

            #[divan::bench]
            fn zip_numcast(bencher: Bencher) {
                let (values, mask) = inputs();
                bencher.bench(|| k_zip_numcast::<$F, $T>(values.as_slice(), &mask));
            }

            #[divan::bench]
            fn zip_select(bencher: Bencher) {
                let (values, mask) = inputs();
                bencher.bench(|| k_zip_select::<$F, $T>(values.as_slice(), &mask));
            }

            #[divan::bench]
            fn bulk_mul(bencher: Bencher) {
                let (values, mask) = inputs();
                bencher.bench(|| k_bulk_mul::<$F, $T>(values.as_slice(), &mask));
            }

            #[divan::bench]
            fn bulk_select(bencher: Bencher) {
                let (values, mask) = inputs();
                bencher.bench(|| k_bulk_select::<$F, $T>(values.as_slice(), &mask));
            }

            // Same-type masking only (`$F -> $F`, no cast). `copy` is the floor.
            #[divan::bench]
            fn copy(bencher: Bencher) {
                let (values, _) = inputs();
                bencher.bench(|| k_copy::<$F>(values.as_slice()));
            }

            #[divan::bench]
            fn bitand(bencher: Bencher) {
                let (values, mask) = inputs();
                bencher.bench(|| k_bitand::<$F>(values.as_slice(), &mask));
            }

            #[divan::bench]
            fn bitand_fused(bencher: Bencher) {
                let (values, mask) = inputs();
                bencher.bench(|| k_bitand_fused::<$F>(values.as_slice(), &mask));
            }

            #[divan::bench]
            fn bitand_words(bencher: Bencher) {
                let (values, mask) = inputs();
                bencher.bench(|| k_bitand_words::<$F>(values.as_slice(), &mask));
            }

            #[divan::bench]
            fn bitand_blocked(bencher: Bencher) {
                let (values, mask) = inputs();
                bencher.bench(|| k_bitand_blocked::<$F>(values.as_slice(), &mask));
            }
        }
    };
}

bench_pair!(i32_to_i32, i32, i32);
bench_pair!(f32_to_f32, f32, f32);
bench_pair!(i64_to_i32, i64, i32);

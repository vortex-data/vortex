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

use divan::Bencher;
use num_traits::AsPrimitive;
use num_traits::NumCast;
use rand::prelude::*;
use vortex_array::dtype::NativePType;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

fn main() {
    divan::main();
}

const N: usize = 100_000;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Baseline: pure cast, ignores validity. Lower bound on achievable time.
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
        }
    };
}

bench_pair!(i32_to_i32, i32, i32);
bench_pair!(f32_to_f32, f32, f32);
bench_pair!(i64_to_i32, i64, i32);

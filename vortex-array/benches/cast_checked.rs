// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for a *checked* primitive cast `F -> T` where an out-of-range
//! value is an error **only if the element is valid**. Invalid (undef)
//! positions may hold out-of-range garbage and must be ignored.
//!
//! The error condition is `OR_i (lossy_i AND valid_i)`, where
//! `lossy_i = (values[i] as T) as F != values[i]`.
//!
//! The kernels below show the progression:
//! - `cast_only`: the cast with no check at all (the floor).
//! - `cast_checked_naive`: the literal per-element zip of values with
//!   `BitBuffer::iter()`. Consuming validity one bit at a time does not
//!   vectorize and is the slow path.
//! - `cast_checked_bitmap`: cast and build a packed `lossy` bitmap in one
//!   vectorized pass (a SIMD compare lands in a mask register, `kmov` extracts
//!   the word), then apply "only if valid" as a word-wise `lossy & validity`
//!   AND. Fully vectorized, safe, no per-element bitmask consumption.
//!
//! When the cast cannot lose data (same width or widening) `lossy` is provably
//! false, so the whole check is eliminated as dead code and every kernel
//! collapses to a plain vectorized cast. The interesting case is narrowing
//! (e.g. `i64 -> i32`).

use divan::Bencher;
use num_traits::AsPrimitive;
use vortex_array::dtype::NativePType;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

fn main() {
    divan::main();
}

const N: usize = 100_000;

/// Floor: cast every element, no validity check.
#[inline(never)]
fn cast_only<F, T>(values: &[F]) -> Buffer<T>
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    BufferMut::from_trusted_len_iter(values.iter().map(|&v| v.as_())).freeze()
}

/// Literal per-element zip of values with the validity bit iterator. Correct,
/// but `BitBuffer::iter()` (one bool per bit via shift+extract) blocks
/// vectorization of the narrowing case.
#[inline(never)]
fn cast_checked_naive<F, T>(values: &[F], validity: &BitBuffer) -> (Buffer<T>, bool)
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType + AsPrimitive<F>,
{
    let mut err = false;
    let out = BufferMut::from_trusted_len_iter(values.iter().zip(validity.iter()).map(
        |(&v, valid)| {
            let t: T = v.as_();
            err |= valid && !(<T as AsPrimitive<F>>::as_(t)).is_eq(v);
            t
        },
    ))
    .freeze();
    (out, err)
}

/// One vectorized pass: cast and build a packed 64-bit `lossy` word per 64
/// elements, then apply "only if valid" as a word-wise `lossy & validity` AND.
/// Validity is consumed 64 bits at a time, never one bit at a time.
///
/// Assumes `validity.offset() == 0` (true for the data this bench constructs).
#[inline(never)]
fn cast_checked_bitmap<F, T>(values: &[F], validity: &BitBuffer) -> (Buffer<T>, bool)
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType + AsPrimitive<F>,
{
    let len = values.len();
    let mut out = BufferMut::<T>::zeroed(len);
    let dst = out.as_mut_slice();
    let vbytes = validity.inner().as_slice();
    let full = len / 64;
    let mut bad = 0u64;
    for (blk, (vchunk, ochunk)) in values[..full * 64]
        .chunks_exact(64)
        .zip(dst.chunks_exact_mut(64))
        .enumerate()
    {
        let mut lossy = 0u64;
        for (j, (o, &v)) in ochunk.iter_mut().zip(vchunk).enumerate() {
            let t: T = v.as_();
            *o = t;
            let lost = !(<T as AsPrimitive<F>>::as_(t)).is_eq(v);
            lossy |= (lost as u64) << j;
        }
        let mut vb8 = [0u8; 8];
        vb8.copy_from_slice(&vbytes[blk * 8..blk * 8 + 8]);
        bad |= lossy & u64::from_le_bytes(vb8);
    }
    let mut err = bad != 0;
    for i in (full * 64)..len {
        let v = values[i];
        let t: T = v.as_();
        dst[i] = t;
        let valid = (vbytes[i >> 3] >> (i & 7)) & 1 == 1;
        err |= valid && !(<T as AsPrimitive<F>>::as_(t)).is_eq(v);
    }
    (out.freeze(), err)
}

fn inputs<F>() -> (Buffer<F>, BitBuffer)
where
    F: NativePType,
    u16: AsPrimitive<F>,
{
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(42);
    let values = BufferMut::from_trusted_len_iter((0..N).map(|_| {
        let v: u16 = rng.random();
        v.as_()
    }))
    .freeze();
    let validity = BitBuffer::from((0..N).map(|_| rng.random_bool(0.5)).collect::<Vec<bool>>());
    (values, validity)
}

macro_rules! bench_pair {
    ($mod_name:ident, $F:ty, $T:ty) => {
        mod $mod_name {
            use super::*;

            #[divan::bench]
            fn cast_only(bencher: Bencher) {
                let (values, _) = inputs::<$F>();
                bencher.bench(|| super::cast_only::<$F, $T>(values.as_slice()));
            }

            #[divan::bench]
            fn cast_checked_naive(bencher: Bencher) {
                let (values, validity) = inputs::<$F>();
                bencher.bench(|| super::cast_checked_naive::<$F, $T>(values.as_slice(), &validity));
            }

            #[divan::bench]
            fn cast_checked_bitmap(bencher: Bencher) {
                let (values, validity) = inputs::<$F>();
                bencher.bench(|| super::cast_checked_bitmap::<$F, $T>(values.as_slice(), &validity));
            }
        }
    };
}

bench_pair!(i32_to_i32, i32, i32);
bench_pair!(f32_to_f32, f32, f32);
bench_pair!(i64_to_i32, i64, i32);

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The segmented head/tail decode of `src/fill.rs`, written against fearless_simd vectors and
//! dispatched to the best instruction set detected at runtime.
//!
//! The algorithm, head widths and kernel thresholds match `src/fill.rs`. The differences are the
//! runtime dispatch, which lets a portable build store heads with AVX2 or AVX-512, and that every
//! chunked fill uses whole head-width vectors.

// The helpers must inline into the dispatched `#[simd]` function to run with its target features.
#![expect(clippy::inline_always)]

use std::mem::MaybeUninit;

use fearless_simd::Level;
use fearless_simd::Simd;
use fearless_simd::dispatch;
use fearless_simd::prelude::*;
use fearless_simd::u8x32;
use fearless_simd::u16x32;
use fearless_simd::u32x16;
use fearless_simd::u64x8;
use fearless_simd_macros::simd;
use vortex_buffer::BufferMut;

const SEGMENT_RUNS: usize = 1024;
const LONG_RUN: usize = 1024;

/// A value type with a fearless_simd vector of one head chunk.
pub trait Lane: Copy {
    /// `[Self; HEAD]`.
    type Head: Copy;
    const HEAD: usize;
    /// Segments with a shorter average run take the head/tail kernel.
    const MAX_AVG_RUN: usize;

    fn splat<S: Simd>(simd: S, value: Self) -> Self::Head;
}

macro_rules! lane {
    ($t:ty, $vector:ident, $head:literal, $max_avg_run:literal) => {
        impl Lane for $t {
            type Head = [$t; $head];
            const HEAD: usize = $head;
            const MAX_AVG_RUN: usize = $max_avg_run;

            #[inline(always)]
            fn splat<S: Simd>(simd: S, value: Self) -> Self::Head {
                $vector::splat(simd, value).into()
            }
        }
    };
}

lane!(u8, u8x32, 32, 24);
lane!(u16, u16x32, 32, 40);
lane!(u32, u32x16, 16, 60);
lane!(u64, u64x8, 8, 25);

pub fn decode_runs<T: Lane>(runs: impl Iterator<Item = (usize, T)>, length: usize) -> BufferMut<T> {
    dispatch!(Level::new(), simd => decode_segments(simd, runs, length))
}

#[simd]
fn decode_segments<S: Simd, T: Lane, I: Iterator<Item = (usize, T)>>(
    simd: S,
    mut runs: I,
    length: usize,
) -> BufferMut<T> {
    let mut decoded = BufferMut::<T>::with_capacity(length);
    let dst = decoded.as_mut_ptr();
    let mut ends = [MaybeUninit::<usize>::uninit(); SEGMENT_RUNS];
    let mut values = [MaybeUninit::<T>::uninit(); SEGMENT_RUNS];
    let mut pos = 0;
    loop {
        let segment_start = pos;
        let mut count = 0;
        for (end, value) in runs.by_ref().take(SEGMENT_RUNS) {
            assert!(end >= pos && end <= length, "invalid run end {end}");
            ends[count].write(end);
            values[count].write(value);
            pos = end;
            count += 1;
        }
        if count == 0 {
            break;
        }
        // SAFETY: the loop above initialized the first `count` ends and values.
        let (ends, values) =
            unsafe { (assume_init(&ends[..count]), assume_init(&values[..count])) };
        // SAFETY: the ends lie in `[segment_start, length]` and are non-decreasing, and the
        // buffer has `length` capacity.
        unsafe {
            if pos - segment_start < T::MAX_AVG_RUN * count {
                fill_head_tail(simd, dst, ends, values, segment_start, length);
            } else {
                let mut run_start = segment_start;
                for (&end, &value) in ends.iter().zip(values) {
                    fill_exact(simd, dst.add(run_start), value, end - run_start);
                    run_start = end;
                }
            }
        }
        if count < SEGMENT_RUNS {
            break;
        }
    }
    // SAFETY: the segments filled every element of `[0, pos)`.
    unsafe { decoded.set_len(pos) };
    decoded
}

/// # Safety
///
/// Every element of `slice` must be initialized.
#[inline]
unsafe fn assume_init<T>(slice: &[MaybeUninit<T>]) -> &[T] {
    // SAFETY: `MaybeUninit<T>` has the layout of `T`, and the caller guarantees initialization.
    unsafe { &*(std::ptr::from_ref(slice) as *const [T]) }
}

/// # Safety
///
/// As `fill_head_tail` in `src/fill.rs`.
#[inline(always)]
unsafe fn fill_head_tail<S: Simd, T: Lane>(
    simd: S,
    dst: *mut T,
    ends: &[usize],
    values: &[T],
    start: usize,
    length: usize,
) {
    let fast = if start + T::HEAD > length {
        0
    } else {
        (ends.partition_point(|&end| end + T::HEAD <= length) + 1).min(ends.len())
    };

    let mut long = [MaybeUninit::<u16>::uninit(); SEGMENT_RUNS];
    let mut long_count = 0;
    let mut pos = start;
    for (i, (&end, &value)) in ends[..fast].iter().zip(&values[..fast]).enumerate() {
        // SAFETY: `pos + HEAD <= length` for the first `fast` runs.
        unsafe {
            dst.add(pos)
                .cast::<T::Head>()
                .write_unaligned(T::splat(simd, value));
        }
        #[expect(clippy::cast_possible_truncation, reason = "i < SEGMENT_RUNS")]
        long[long_count].write(i as u16);
        long_count += usize::from(end - pos > T::HEAD);
        pos = end;
    }

    // SAFETY: the loop above wrote `long[k]` for every `k` it advanced past.
    for &i in unsafe { assume_init(&long[..long_count]) } {
        let i = usize::from(i);
        let run_start = if i == 0 { start } else { ends[i - 1] };
        let n = ends[i] - run_start;
        // SAFETY: run `i` covers `[run_start, ends[i])`, which is longer than `HEAD`.
        unsafe {
            if n > LONG_RUN {
                fill_exact(simd, dst.add(run_start), values[i], n);
            } else {
                fill_chunks(simd, dst.add(run_start), values[i], T::HEAD, n);
            }
        }
    }

    for (&end, &value) in ends[fast..].iter().zip(&values[fast..]) {
        // SAFETY: `pos <= end <= length`.
        unsafe { fill_exact(simd, dst.add(pos), value, end - pos) };
        pos = end;
    }
}

/// Writes `value` to `dst[..n]` and nowhere else.
///
/// # Safety
///
/// `dst` must be valid for writes of `n` elements.
#[inline(always)]
unsafe fn fill_exact<S: Simd, T: Lane>(simd: S, dst: *mut T, value: T, n: usize) {
    // SAFETY: forwarded from the caller.
    unsafe {
        if size_of::<T>() == 1 {
            std::ptr::write_bytes(
                dst.cast::<u8>(),
                std::mem::transmute_copy::<T, u8>(&value),
                n,
            );
        } else if n < T::HEAD {
            for i in 0..n {
                dst.add(i).write(value);
            }
        } else {
            fill_chunks(simd, dst, value, 0, n);
        }
    }
}

/// Fills `dst[written..n]` with head-width vectors, the last overlapping the previous one to end
/// exactly at `n`.
///
/// # Safety
///
/// `HEAD <= n`, `written <= n`, and `dst` must be valid for writes of `n` elements.
#[inline(always)]
unsafe fn fill_chunks<S: Simd, T: Lane>(
    simd: S,
    dst: *mut T,
    value: T,
    mut written: usize,
    n: usize,
) {
    let chunk = T::splat(simd, value);
    // SAFETY: every store lies in `[0, n)`.
    unsafe {
        while n - written > T::HEAD {
            dst.add(written).cast::<T::Head>().write_unaligned(chunk);
            written += T::HEAD;
        }
        dst.add(n - T::HEAD)
            .cast::<T::Head>()
            .write_unaligned(chunk);
    }
}

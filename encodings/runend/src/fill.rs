// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expanding runs into a flat buffer.
//!
//! In real columns short and long runs interleave almost at random, so any per-run decision on the
//! run length mispredicts about as often as it is taken. Runs are therefore decoded a segment of
//! [`SEGMENT_RUNS`] runs at a time, and the kernel is chosen once per segment from the segment's
//! average run length:
//!
//! - Short runs on average: every run takes one fixed-width "head" store of up to 64 bytes without
//!   a branch, and only the runs longer than the head are revisited to fill their tails.
//! - Long runs on average: one fill per run (a `memset` for bytes), whose cost the run amortizes.

use std::mem::MaybeUninit;

use vortex_buffer::BufferMut;

/// Runs decoded per segment, and so how often the kernel can change.
const SEGMENT_RUNS: usize = 1024;

/// Tails of runs longer than this are filled by a plain loop rather than head-sized chunks.
const LONG_RUN: usize = 1024;

/// Decodes `(end, value)` runs into a buffer of `length` elements.
///
/// Ends must be non-decreasing and at most `length`; the returned buffer is as long as the last
/// end.
pub(crate) fn decode_runs<T: Copy>(
    runs: impl Iterator<Item = (usize, T)>,
    length: usize,
) -> BufferMut<T> {
    // Head chunks of 32 bytes for bytes and 64 bytes otherwise, and the average run length below
    // which the head/tail kernel wins, as measured on ClickBench and TPC-DS columns.
    match size_of::<T>() {
        1 => decode_segments::<T, 32>(runs, length, 24),
        2 => decode_segments::<T, 32>(runs, length, 40),
        4 => decode_segments::<T, 16>(runs, length, 60),
        8 => decode_segments::<T, 8>(runs, length, 25),
        _ => decode_exact(runs, length),
    }
}

/// Fills every run exactly, for values too wide to gain from head stores.
fn decode_exact<T: Copy>(runs: impl Iterator<Item = (usize, T)>, length: usize) -> BufferMut<T> {
    let mut decoded = BufferMut::<T>::with_capacity(length);
    let mut pos = 0;
    for (end, value) in runs {
        check_end(end, pos, length);
        // SAFETY: `pos <= end <= length` and the buffer has `length` capacity.
        unsafe { decoded.push_n_unchecked(value, end - pos) };
        pos = end;
    }
    // SAFETY: the runs filled every element of `[0, pos)`.
    unsafe { decoded.set_len(pos) };
    decoded
}

/// Decodes runs a segment at a time with head chunks of `C` elements, taking the head/tail kernel
/// for segments whose average run is shorter than `max_avg_run`.
fn decode_segments<T: Copy, const C: usize>(
    mut runs: impl Iterator<Item = (usize, T)>,
    length: usize,
    max_avg_run: usize,
) -> BufferMut<T> {
    let mut decoded = BufferMut::<T>::with_capacity(length);
    let dst = decoded.as_mut_ptr();
    // Left uninitialized: zeroing them would cost as much as decoding a small array.
    let mut ends = [MaybeUninit::<usize>::uninit(); SEGMENT_RUNS];
    let mut values = [MaybeUninit::<T>::uninit(); SEGMENT_RUNS];
    let mut pos = 0;
    loop {
        let segment_start = pos;
        let mut count = 0;
        for (end, value) in runs.by_ref().take(SEGMENT_RUNS) {
            check_end(end, pos, length);
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
        let rows = pos - segment_start;
        // SAFETY: the ends lie in `[segment_start, length]` and are non-decreasing, and the
        // buffer has `length` capacity.
        unsafe {
            if rows < max_avg_run * count {
                fill_head_tail::<T, C>(dst, ends, values, segment_start, length);
            } else {
                fill_each(dst, ends, values, segment_start);
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

/// Views initialized elements as a plain slice.
///
/// # Safety
///
/// Every element of `slice` must be initialized.
#[inline]
unsafe fn assume_init<T>(slice: &[MaybeUninit<T>]) -> &[T] {
    // SAFETY: `MaybeUninit<T>` has the layout of `T`, and the caller guarantees initialization.
    unsafe { &*(std::ptr::from_ref(slice) as *const [T]) }
}

#[inline]
fn check_end(end: usize, pos: usize, length: usize) {
    assert!(
        end >= pos,
        "Runend ends must be monotonic, got {end} after {pos}"
    );
    assert!(end <= length, "Runend end must be less than overall length");
}

/// Fills a segment run by run.
///
/// # Safety
///
/// `ends` must be non-decreasing, start at or after `start`, and `dst` must be valid for writes
/// up to the last end.
unsafe fn fill_each<T: Copy>(dst: *mut T, ends: &[usize], values: &[T], start: usize) {
    let mut pos = start;
    for (&end, &value) in ends.iter().zip(values) {
        // SAFETY: `pos <= end`, within the caller's bounds.
        unsafe { fill_exact(dst.add(pos), value, end - pos) };
        pos = end;
    }
}

/// Fills a segment of mostly short runs in two passes.
///
/// The first pass stores a head chunk of `C` elements at the start of every run without a branch
/// on the run length, and lists the runs longer than `C`. A head chunk may reach past its run;
/// those elements belong to later runs, which are written afterwards and overwrite them, so the
/// final contents are exact. Heads are never stored past `length`: the last runs, whose head would
/// cross it, are filled exactly. The second pass fills the tails of the listed runs.
///
/// # Safety
///
/// `ends` must be non-decreasing, start at or after `start`, end at most `length`, and `dst` must
/// be valid for writes of `length` elements.
unsafe fn fill_head_tail<T: Copy, const C: usize>(
    dst: *mut T,
    ends: &[usize],
    values: &[T],
    start: usize,
    length: usize,
) {
    // Run `i` starts at the previous end, so its head fits when that end is at most
    // `length - C`. Ends are sorted, so these runs form a prefix.
    let fast = if start + C > length {
        0
    } else {
        (ends.partition_point(|&end| end + C <= length) + 1).min(ends.len())
    };

    let mut long = [MaybeUninit::<u16>::uninit(); SEGMENT_RUNS];
    let mut long_count = 0;
    let mut pos = start;
    for (i, (&end, &value)) in ends[..fast].iter().zip(&values[..fast]).enumerate() {
        // SAFETY: `pos + C <= length` for the first `fast` runs.
        unsafe { dst.add(pos).cast::<[T; C]>().write_unaligned([value; C]) };
        // Record every run and only advance past the long ones, which avoids a branch.
        // `i < SEGMENT_RUNS` fits a u16.
        #[allow(clippy::cast_possible_truncation)]
        {
            long[long_count].write(i as u16);
        }
        long_count += usize::from(end - pos > C);
        pos = end;
    }

    // SAFETY: the loop above wrote `long[k]` for every `k` it advanced past.
    for &i in unsafe { assume_init(&long[..long_count]) } {
        let i = usize::from(i);
        let run_start = if i == 0 { start } else { ends[i - 1] };
        // SAFETY: run `i` covers `[run_start, ends[i])`, which is longer than `C`.
        unsafe { fill_tail::<T, C>(dst.add(run_start), values[i], ends[i] - run_start) };
    }

    for (&end, &value) in ends[fast..].iter().zip(&values[fast..]) {
        // SAFETY: `pos <= end <= length`.
        unsafe { fill_exact(dst.add(pos), value, end - pos) };
        pos = end;
    }
}

/// Fills `dst[C..n]` of a run of `n > C` elements whose head chunk is already stored, with
/// `C`-element chunks and a last chunk that overlaps the previous one to end exactly at `n`.
///
/// # Safety
///
/// `n > C` and `dst` must be valid for writes of `n` elements.
#[inline]
unsafe fn fill_tail<T: Copy, const C: usize>(dst: *mut T, value: T, n: usize) {
    // SAFETY: every store lies in `[C, n)`, or `[0, n)` for long runs.
    unsafe {
        if n > LONG_RUN {
            fill_exact(dst, value, n);
            return;
        }
        let chunk = [value; C];
        let mut written = C;
        while n - written > C {
            dst.add(written).cast::<[T; C]>().write_unaligned(chunk);
            written += C;
        }
        dst.add(n - C).cast::<[T; C]>().write_unaligned(chunk);
    }
}

/// Writes `value` to `dst[..n]` and nowhere else.
///
/// # Safety
///
/// `dst` must be valid for writes of `n` elements.
#[inline]
unsafe fn fill_exact<T: Copy>(dst: *mut T, value: T, n: usize) {
    // SAFETY: forwarded from the caller.
    unsafe {
        if size_of::<T>() == 1 {
            // glibc's `memset` sizes in bytes and is hard to beat per run.
            std::ptr::write_bytes(
                dst.cast::<u8>(),
                std::mem::transmute_copy::<T, u8>(&value),
                n,
            );
        } else if size_of::<T>() >= 8 {
            // Keep chunks at or below 32 bytes for wide values: larger overlapping stores cost
            // more than they save.
            fill_exact_chunked::<T, 4>(dst, value, n);
        } else {
            fill_exact_chunked::<T, 8>(dst, value, n);
        }
    }
}

/// Runs longer than this are filled by a plain loop in [`fill_exact_chunked`].
const PLAIN_RUN: usize = 64;

/// [`fill_exact`] with chunks of `CHUNK` elements, where `CHUNK` is 4 or 8: short runs take a pair
/// of overlapping stores, medium runs whole chunks with a last chunk overlapping the previous one
/// to end exactly at `n`.
///
/// # Safety
///
/// `dst` must be valid for writes of `n` elements.
#[inline]
unsafe fn fill_exact_chunked<T: Copy, const CHUNK: usize>(dst: *mut T, value: T, n: usize) {
    // SAFETY: every store below lies within `dst[..n]`: pairs of stores cover `[0, n)` from
    // both ends, chunks in the loop end before `n`, and the last chunk ends at `n`.
    unsafe {
        if n < CHUNK {
            if CHUNK > 4 && n >= 4 {
                dst.cast::<[T; 4]>().write_unaligned([value; 4]);
                dst.add(n - 4).cast::<[T; 4]>().write_unaligned([value; 4]);
            } else if n >= 2 {
                dst.cast::<[T; 2]>().write_unaligned([value; 2]);
                dst.add(n - 2).cast::<[T; 2]>().write_unaligned([value; 2]);
            } else if n == 1 {
                dst.write(value);
            }
            return;
        }
        if n > PLAIN_RUN {
            // Long runs amortize the loop exit, and the compiler widens this loop beyond what
            // `CHUNK` stores manage.
            for i in 0..n {
                dst.add(i).write(value);
            }
            return;
        }
        let chunk = [value; CHUNK];
        let mut written = 0;
        while n - written > CHUNK {
            dst.add(written).cast::<[T; CHUNK]>().write_unaligned(chunk);
            written += CHUNK;
        }
        dst.add(n - CHUNK)
            .cast::<[T; CHUNK]>()
            .write_unaligned(chunk);
    }
}

#[cfg(test)]
// Run values only need to differ from their neighbours, so wrapping casts are fine.
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;

    use super::SEGMENT_RUNS;
    use super::decode_runs;

    /// Expands `(length, value)` runs one element at a time.
    fn expand<T: Copy>(runs: &[(usize, T)]) -> Vec<T> {
        runs.iter()
            .flat_map(|&(len, value)| std::iter::repeat_n(value, len))
            .collect()
    }

    fn check<T: Copy + PartialEq + std::fmt::Debug>(runs: &[(usize, T)]) {
        let expected = expand(runs);
        let mut end = 0;
        let ends = runs.iter().map(|&(len, value)| {
            end += len;
            (end, value)
        });
        let decoded = decode_runs(ends, expected.len());
        assert_eq!(decoded.as_slice(), expected.as_slice());
    }

    /// Run lengths drawn per segment from short, long, or mixed distributions, so segments take
    /// both kernels, plus zero-length runs.
    fn lengths(seed: u64, count: usize) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..count)
            .map(|i| match (i / SEGMENT_RUNS) % 3 {
                0 => rng.random_range(0..=9),
                1 => rng.random_range(50..=3000),
                _ => {
                    if rng.random_bool(0.5) {
                        1
                    } else {
                        rng.random_range(0..=200)
                    }
                }
            })
            .collect()
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(2)]
    fn decode_runs_matches_expansion(#[case] seed: u64) {
        let lens = lengths(seed, 5 * SEGMENT_RUNS + 17);
        check::<u8>(
            &lens
                .iter()
                .enumerate()
                .map(|(i, &l)| (l, i as u8))
                .collect::<Vec<_>>(),
        );
        check::<u16>(
            &lens
                .iter()
                .enumerate()
                .map(|(i, &l)| (l, i as u16))
                .collect::<Vec<_>>(),
        );
        check::<u32>(
            &lens
                .iter()
                .enumerate()
                .map(|(i, &l)| (l, i as u32))
                .collect::<Vec<_>>(),
        );
        check::<u64>(
            &lens
                .iter()
                .enumerate()
                .map(|(i, &l)| (l, i as u64))
                .collect::<Vec<_>>(),
        );
        check::<u128>(
            &lens
                .iter()
                .enumerate()
                .map(|(i, &l)| (l, i as u128))
                .collect::<Vec<_>>(),
        );
    }

    /// Short runs right up to the end of the buffer, where head chunks no longer fit.
    #[rstest]
    #[case(1)]
    #[case(7)]
    #[case(31)]
    #[case(33)]
    fn decode_runs_short_tail(#[case] total_runs: usize) {
        let runs: Vec<(usize, u8)> = (0..total_runs).map(|i| (1 + i % 2, i as u8)).collect();
        check(&runs);
        let runs: Vec<(usize, u64)> = (0..total_runs).map(|i| (1 + i % 2, i as u64)).collect();
        check(&runs);
    }

    /// Run lengths around each head width, in a segment that takes the head kernel.
    #[test]
    fn decode_runs_around_head_width() {
        let runs: Vec<(usize, u32)> = (0..2000u32)
            .map(|i| {
                (
                    match i % 6 {
                        0 => 15,
                        1 => 16,
                        2 => 17,
                        3 => 32,
                        4 => 33,
                        _ => 1,
                    },
                    i,
                )
            })
            .collect();
        check(&runs);
        let runs: Vec<(usize, u8)> = (0..2000usize)
            .map(|i| {
                (
                    match i % 5 {
                        0 => 31,
                        1 => 32,
                        2 => 33,
                        3 => 64,
                        _ => 1,
                    },
                    i as u8,
                )
            })
            .collect();
        check(&runs);
    }

    #[test]
    fn decode_runs_empty() {
        check::<u32>(&[]);
        check::<u32>(&[(0, 1), (0, 2)]);
    }
}

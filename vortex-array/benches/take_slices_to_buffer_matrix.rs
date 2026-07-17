// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for primitive `take_slices_to_buffer` copy-loop variants.
//!
//! The matrix covers:
//! - append via `BufferMut::extend_from_slice` vs direct cursor copy into spare output capacity
//! - ordinary checked slicing vs a preverification pass followed by unchecked slicing

#![allow(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use divan::Bencher;
use divan::counter::BytesCount;
use rand::SeedableRng;
use rand::prelude::*;
use rand_distr::Distribution;
use rand_distr::Normal;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

fn main() {
    divan::main();
}

const SOURCE_LENS: &[usize] = &[1_000, 10_000, 100_000];
const SLICE_COUNT: usize = 50;

type TakeSlicesFn = fn(&[u16], &[usize], &[usize], usize) -> Buffer<u16>;

#[derive(Clone)]
struct Case {
    values: Vec<u16>,
    starts: Vec<usize>,
    lengths: Vec<usize>,
    output_len: usize,
}

#[divan::bench(args = SOURCE_LENS)]
fn extend_safe(bencher: Bencher, source_len: usize) {
    let case = Case::new(source_len);
    bench_case(bencher, case, take_extend_safe);
}

#[divan::bench(args = SOURCE_LENS)]
fn cursor_copy_safe(bencher: Bencher, source_len: usize) {
    let case = Case::new(source_len);
    bench_case(bencher, case, take_cursor_copy_safe);
}

#[divan::bench(args = SOURCE_LENS)]
fn preverify_extend_unchecked(bencher: Bencher, source_len: usize) {
    let case = Case::new(source_len);
    bench_case(bencher, case, take_preverify_extend_unchecked);
}

#[divan::bench(args = SOURCE_LENS)]
fn preverify_cursor_copy_unchecked(bencher: Bencher, source_len: usize) {
    let case = Case::new(source_len);
    bench_case(bencher, case, take_preverify_cursor_copy_unchecked);
}

fn bench_case(bencher: Bencher, case: Case, f: TakeSlicesFn) {
    bencher
        .counter(BytesCount::of_many::<u16>(case.output_len))
        .with_inputs(|| (&case.values, &case.starts, &case.lengths, case.output_len))
        .bench_refs(|(values, starts, lengths, output_len)| {
            divan::black_box(f(values, starts, lengths, *output_len));
        });
}

impl Case {
    fn new(source_len: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(source_len as u64 ^ 0x5eed_51ce);
        let values = (0..source_len).map(|_| rng.random::<u16>()).collect();
        let mean = source_len as f64 / 100.0;
        let normal = Normal::new(mean, (mean / 4.0).max(1.0)).unwrap();

        let mut ranges = (0..SLICE_COUNT)
            .map(|_| {
                let length = normal
                    .sample(&mut rng)
                    .round()
                    .clamp(1.0, source_len as f64) as usize;
                let start = rng.random_range(0..=source_len - length);
                (start, length)
            })
            .collect::<Vec<_>>();
        ranges.sort_unstable_by_key(|&(start, _)| start);

        let output_len = ranges.iter().map(|&(_, length)| length).sum();
        let (starts, lengths) = ranges.into_iter().unzip();
        Self {
            values,
            starts,
            lengths,
            output_len,
        }
    }
}

fn take_extend_safe(
    values: &[u16],
    starts: &[usize],
    lengths: &[usize],
    output_len: usize,
) -> Buffer<u16> {
    let mut result = BufferMut::<u16>::with_capacity(output_len);
    for (&start, &length) in starts.iter().zip(lengths) {
        result.extend_from_slice(&values[start..start + length]);
    }
    result.freeze()
}

fn take_cursor_copy_safe(
    values: &[u16],
    starts: &[usize],
    lengths: &[usize],
    output_len: usize,
) -> Buffer<u16> {
    let mut result = BufferMut::<u16>::with_capacity(output_len);
    let mut cursor = 0usize;
    for (&start, &length) in starts.iter().zip(lengths) {
        copy_to_spare(&mut result, cursor, &values[start..start + length]);
        cursor += length;
    }

    // SAFETY: the loop writes exactly `output_len` values into spare capacity.
    unsafe { result.set_len(output_len) };
    result.freeze()
}

fn take_preverify_extend_unchecked(
    values: &[u16],
    starts: &[usize],
    lengths: &[usize],
    output_len: usize,
) -> Buffer<u16> {
    preverify(values.len(), starts, lengths);

    let mut result = BufferMut::<u16>::with_capacity(output_len);
    for (&start, &length) in starts.iter().zip(lengths) {
        // SAFETY: `preverify` checked every source range.
        unsafe {
            result.extend_from_slice(values.get_unchecked(start..start + length));
        }
    }
    result.freeze()
}

fn take_preverify_cursor_copy_unchecked(
    values: &[u16],
    starts: &[usize],
    lengths: &[usize],
    output_len: usize,
) -> Buffer<u16> {
    preverify(values.len(), starts, lengths);

    let mut result = BufferMut::<u16>::with_capacity(output_len);
    let mut cursor = 0usize;
    for (&start, &length) in starts.iter().zip(lengths) {
        // SAFETY: `preverify` checked every source range.
        let source = unsafe { values.get_unchecked(start..start + length) };
        unsafe { copy_to_spare_unchecked(&mut result, cursor, source) };
        cursor += length;
    }

    // SAFETY: the loop writes exactly `output_len` values into spare capacity.
    unsafe { result.set_len(output_len) };
    result.freeze()
}

fn preverify(source_len: usize, starts: &[usize], lengths: &[usize]) {
    for (&start, &length) in starts.iter().zip(lengths) {
        let end = start.checked_add(length).unwrap();
        assert!(end <= source_len);
    }
}

fn copy_to_spare(result: &mut BufferMut<u16>, cursor: usize, source: &[u16]) {
    let dst = &mut result.spare_capacity_mut()[cursor..][..source.len()];
    // SAFETY: `dst` has exactly `source.len()` spare slots and does not overlap with source.
    unsafe { copy_to_uninit(dst.as_mut_ptr().cast(), source) };
}

unsafe fn copy_to_spare_unchecked(result: &mut BufferMut<u16>, cursor: usize, source: &[u16]) {
    // SAFETY: callers ensure `cursor..cursor + source.len()` is within spare capacity.
    let dst = unsafe {
        result
            .spare_capacity_mut()
            .get_unchecked_mut(cursor..cursor + source.len())
    };
    // SAFETY: `dst` has exactly `source.len()` spare slots and does not overlap with source.
    unsafe { copy_to_uninit(dst.as_mut_ptr().cast(), source) };
}

unsafe fn copy_to_uninit(dst: *mut u16, source: &[u16]) {
    // SAFETY: callers ensure `dst` points to `source.len()` writable uninitialized u16 slots.
    unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), dst, source.len()) };
}

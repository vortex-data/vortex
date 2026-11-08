// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use fastlanes::Delta;
use fastlanes::Transpose;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;

use super::U8_LANES;
use super::gather_chunk;
use super::gather_chunk_scalar;
use super::gather_delta_lanes;
use super::gather_delta_lanes_scalar;
use crate::FL_CHUNK_SIZE;

/// RLE indices for one chunk: each step is 0 or 1, with runs of `1..=max_run`.
fn rle_indices(rng: &mut StdRng, max_run: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(FL_CHUNK_SIZE);
    let mut index = 0;
    while indices.len() < FL_CHUNK_SIZE {
        let run = rng.random_range(1..=max_run);
        indices.extend(std::iter::repeat_n(index, run));
        index += 1;
    }
    indices.truncate(FL_CHUNK_SIZE);
    indices
}

fn values<V: TryFrom<u64>>(len: usize) -> Vec<V> {
    (0..len as u64)
        .map(|i| V::try_from(i * 37 % 127 + 1).ok().unwrap())
        .collect()
}

fn check_gather<V, I>(indices: &[usize], extra_values: usize)
where
    V: Copy + PartialEq + std::fmt::Debug + TryFrom<u64>,
    I: Copy + Into<usize> + TryFrom<usize>,
{
    let max = indices.iter().copied().max().unwrap();
    let vals = values::<V>(max + 1 + extra_values);
    let typed: Vec<I> = indices
        .iter()
        .map(|&i| I::try_from(i).ok().unwrap())
        .collect();
    let typed: &[I; FL_CHUNK_SIZE] = typed.as_slice().try_into().unwrap();
    let mut out = [vals[0]; FL_CHUNK_SIZE];
    // SAFETY: every index is at most `max`, below `vals.len()`.
    unsafe { gather_chunk(&vals, typed, &mut out) };
    let expected: Vec<V> = indices.iter().map(|&i| vals[i]).collect();
    assert_eq!(out.as_slice(), expected.as_slice());

    let mut out = [vals[0]; FL_CHUNK_SIZE];
    // SAFETY: as above.
    unsafe { gather_chunk_scalar(&vals, typed, &mut out) };
    assert_eq!(out.as_slice(), expected.as_slice());
}

fn check_all_widths(indices: &[usize], extra_values: usize) {
    check_gather::<u8, u16>(indices, extra_values);
    check_gather::<u16, u16>(indices, extra_values);
    check_gather::<u32, u16>(indices, extra_values);
    check_gather::<u64, u16>(indices, extra_values);
    if indices.iter().all(|&i| i <= usize::from(u8::MAX)) {
        check_gather::<u8, u8>(indices, extra_values);
        check_gather::<u16, u8>(indices, extra_values);
        check_gather::<u32, u8>(indices, extra_values);
        check_gather::<u64, u8>(indices, extra_values);
    }
}

#[rstest]
#[case::runs_of_1(1)]
#[case::runs_up_to_4(4)]
#[case::runs_up_to_15(15)]
#[case::runs_up_to_64(64)]
fn gather_matches_scalar(#[case] max_run: usize) {
    let mut rng = StdRng::seed_from_u64(max_run as u64);
    let indices = rle_indices(&mut rng, max_run);
    // With no spare values, the last groups' windows would run past the end and must take the
    // scalar path; with spare values, every group takes the vector path.
    check_all_widths(&indices, 0);
    check_all_widths(&indices, 8);
}

/// Indices that are not RLE indices (spread > 7, or decreasing) still gather correctly.
#[test]
fn gather_handles_any_in_bounds_indices() {
    let mut rng = StdRng::seed_from_u64(3);
    let indices: Vec<usize> = (0..FL_CHUNK_SIZE)
        .map(|_| rng.random_range(0..200))
        .collect();
    check_all_widths(&indices, 0);
}

/// Delta-encode `indices` the way the compressor does for `u8` RLE indices: transposed, with the
/// first row as bases, and bit-pack the deltas at 1 bit.
fn delta_encode(indices: &[usize]) -> ([u8; U8_LANES], [u8; U8_LANES], bool) {
    let input: Vec<u8> = indices.iter().map(|&i| u8::try_from(i).unwrap()).collect();
    let input: &[u8; FL_CHUNK_SIZE] = input.as_slice().try_into().unwrap();
    let mut transposed = [0u8; FL_CHUNK_SIZE];
    u8::transpose(input, &mut transposed);
    let bases: [u8; U8_LANES] = transposed[..U8_LANES].try_into().unwrap();
    let mut deltas = [0u8; FL_CHUNK_SIZE];
    u8::delta::<U8_LANES>(&transposed, &bases, &mut deltas);
    assert!(
        deltas.iter().all(|&d| d <= 1),
        "RLE index deltas are 0 or 1"
    );
    let mut packed = [0u8; U8_LANES];
    // SAFETY: a 1-bit chunk of 1024 `u8` values packs into 128 bytes.
    unsafe { u8::unchecked_pack(1, &deltas, &mut packed) };
    (bases, packed, deltas.iter().all(|&d| d == 0))
}

fn check_delta<V>(indices: &[usize], extra_values: usize)
where
    V: Copy + PartialEq + std::fmt::Debug + TryFrom<u64>,
{
    let (bases, packed, all_zero) = delta_encode(indices);
    let max = indices.iter().copied().max().unwrap();
    let vals = values::<V>(max + 1 + extra_values);
    let expected: Vec<V> = indices.iter().map(|&i| vals[i]).collect();

    let mut out = [vals[0]; FL_CHUNK_SIZE];
    // SAFETY: every lane's last index is an index of the chunk, below `vals.len()`.
    unsafe { gather_delta_lanes(&vals, &bases, Some(&packed), &mut out) };
    assert_eq!(out.as_slice(), expected.as_slice());

    let mut out = [vals[0]; FL_CHUNK_SIZE];
    // SAFETY: as above.
    unsafe { gather_delta_lanes_scalar(&vals, &bases, Some(&packed), &mut out) };
    assert_eq!(out.as_slice(), expected.as_slice());

    if all_zero {
        let mut out = [vals[0]; FL_CHUNK_SIZE];
        // SAFETY: as above; with every delta zero, each lane's last index is its base.
        unsafe { gather_delta_lanes(&vals, &bases, None, &mut out) };
        assert_eq!(out.as_slice(), expected.as_slice());
    }
}

#[rstest]
#[case::runs_of_8(8, true)]
#[case::runs_of_16(16, true)]
#[case::runs_up_to_8(8, false)]
#[case::runs_up_to_15(15, false)]
#[case::runs_up_to_64(64, false)]
fn gather_delta_lanes_matches_scalar(#[case] run: usize, #[case] fixed: bool) {
    let mut rng = StdRng::seed_from_u64(run as u64);
    let indices: Vec<usize> = if fixed {
        (0..FL_CHUNK_SIZE).map(|i| i / run).collect()
    } else {
        rle_indices(&mut rng, run)
    };
    for extra in [0, 8] {
        check_delta::<u8>(&indices, extra);
        check_delta::<u16>(&indices, extra);
        check_delta::<u32>(&indices, extra);
        check_delta::<u64>(&indices, extra);
    }
}

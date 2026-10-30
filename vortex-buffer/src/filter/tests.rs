// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

use std::fmt::Debug;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;

use super::*;

/// The representation of the test selection.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Bitmap,
    Slices,
    Indices,
}

/// A selection of `len` positions, stored at bit `offset` of its bitmap.
fn make_bits(len: usize, offset: usize, pattern: impl Fn(usize) -> bool) -> BitBuffer {
    let backing =
        BitBuffer::from_iter(std::iter::repeat_n(false, offset).chain((0..len).map(pattern)));
    BitBuffer::new_with_offset(backing.inner().clone(), len, offset)
}

fn values<T: From<u8>>(len: usize) -> Vec<T> {
    (0..len).map(|i| T::from((i % 251) as u8)).collect()
}

fn expected<T: Copy>(values: &[T], bits: &BitBuffer) -> Vec<T> {
    values
        .iter()
        .zip(bits.iter())
        .filter_map(|(&value, selected)| selected.then_some(value))
        .collect()
}

/// Filter `values` by `bits` in the representation `kind`, in place when `in_place` is set.
fn filter<T: Copy>(values: &[T], bits: &BitBuffer, kind: Kind, in_place: bool) -> Buffer<T> {
    let slices = bits.set_slices().collect::<Vec<_>>();
    let indices = bits.set_indices().collect::<Vec<_>>();
    // SAFETY: all three representations come from `bits`.
    let selection = unsafe {
        match kind {
            Kind::Bitmap => Selection::from_bitmap_unchecked(bits, bits.true_count()),
            Kind::Slices => {
                Selection::from_slices_unchecked(bits.len(), &slices, bits.true_count())
            }
            Kind::Indices => Selection::from_indices_unchecked(bits.len(), &indices),
        }
    };

    if in_place {
        let mut values = values.to_vec();
        let new_len = filter_slice_in_place(&mut values, selection);
        values.truncate(new_len);
        Buffer::copy_from(values)
    } else {
        filter_slice(values, selection, BufferAllocatorRef::static_ref())
    }
}

/// Compact `values` by `bits` with every scalar bitmap strategy, whatever the ladder chooses.
fn check_bitmap_strategies<T: Copy + PartialEq + Debug>(values: &[T], bits: &BitBuffer) {
    let expected = expected(values, bits);
    let bits = MaskBits::Bitmap(bits);

    let mut out = Vec::with_capacity(values.len());
    // SAFETY: `bits` covers `values`, and the output has room for every value.
    let written = unsafe { scalar::compact_by_bitmap(values, bits, out.spare_capacity_mut()) };
    // SAFETY: the walk initialized the first `written` values.
    unsafe { out.set_len(written) };
    assert_eq!(out, expected);

    let mut out = Vec::with_capacity(values.len());
    let written = scalar::compact_runs_by_bitmap(values, bits, out.spare_capacity_mut());
    // SAFETY: the walk initialized the first `written` values.
    unsafe { out.set_len(written) };
    assert_eq!(out, expected);

    let mut out = Vec::with_capacity(values.len());
    // SAFETY: `bits` covers `values`, and the output has room for every value.
    let written = unsafe { byte_compress::compress(values, bits, out.spare_capacity_mut()) };
    // SAFETY: the walk initialized the first `written` values.
    unsafe { out.set_len(written) };
    assert_eq!(out, expected);

    let mut compacted = values.to_vec();
    let new_len = scalar::compact_in_place_by_bitmap(&mut compacted, bits);
    assert_eq!(&compacted[..new_len], expected.as_slice());
}

fn check<T: Copy + PartialEq + Debug + From<u8>>(bits: &BitBuffer) {
    let values = values::<T>(bits.len());
    let expected = expected(&values, bits);

    for kind in [Kind::Bitmap, Kind::Slices, Kind::Indices] {
        for in_place in [false, true] {
            assert_eq!(
                filter(&values, bits, kind, in_place).as_slice(),
                expected.as_slice(),
                "{kind:?}, in place: {in_place}"
            );
        }
    }

    check_bitmap_strategies(&values, bits);
}

#[rstest]
fn filter_matches_expected(
    #[values(0.001, 0.05, 0.2, 0.5, 0.9, 0.999)] density: f64,
    #[values(0, 3, 64)] offset: usize,
    #[values(9, 151, 2000)] len: usize,
) {
    let mut rng = StdRng::seed_from_u64(42);
    let selected = (0..len)
        .map(|_| rng.random_bool(density))
        .collect::<Vec<_>>();
    let bits = make_bits(len, offset, |i| selected[i]);

    check::<u8>(&bits);
    check::<u16>(&bits);
    check::<u32>(&bits);
    check::<u64>(&bits);
    check::<u128>(&bits);
}

#[rstest]
fn filter_runs_matches_expected(
    #[values(1, 12, 200)] run_len: usize,
    #[values(1, 16, 300)] gap: usize,
    #[values(0, 5)] offset: usize,
) {
    let bits = make_bits(2000, offset, |i| i % (run_len + gap) < run_len);

    check::<u8>(&bits);
    check::<u16>(&bits);
    check::<u32>(&bits);
    check::<u64>(&bits);
    check::<u128>(&bits);
}

#[test]
fn filter_buffer_in_place_and_shared() {
    let bits = make_bits(200, 3, |i| i % 7 != 0);
    let expected = expected(&values::<u32>(200), &bits);
    // SAFETY: the true count comes from `bits`.
    let selection = unsafe { Selection::from_bitmap_unchecked(&bits, bits.true_count()) };

    let unique = Buffer::copy_from(values::<u32>(200));
    let unique_ptr = unique.as_ptr();
    let filtered = filter_buffer(unique, selection, BufferAllocatorRef::static_ref());
    assert_eq!(filtered.as_slice(), expected.as_slice());
    assert_eq!(
        filtered.as_ptr(),
        unique_ptr,
        "a dense unique buffer filters in place"
    );

    let shared = Buffer::copy_from(values::<u32>(200));
    let other = shared.clone();
    let filtered = filter_buffer(shared, selection, BufferAllocatorRef::static_ref());
    assert_eq!(filtered.as_slice(), expected.as_slice());
    assert_eq!(other.as_slice(), values::<u32>(200).as_slice());
}

#[test]
fn filter_buffer_empty() {
    let bits = BitBuffer::from_iter(std::iter::empty::<bool>());
    // SAFETY: an empty bitmap has no set bits.
    let selection = unsafe { Selection::from_bitmap_unchecked(&bits, 0) };

    let filtered = filter_buffer(
        Buffer::<u64>::empty(),
        selection,
        BufferAllocatorRef::static_ref(),
    );
    assert!(filtered.is_empty());
}

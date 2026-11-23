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

/// Call `f` with the selection of `bits` in the representation `kind`.
fn with_selection<R>(bits: &BitBuffer, kind: Kind, f: impl FnOnce(Selection<'_>) -> R) -> R {
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

    f(selection)
}

/// Filter `values` by `bits` in the representation `kind`, in place when `in_place` is set.
fn filter<T: Copy>(values: &[T], bits: &BitBuffer, kind: Kind, in_place: bool) -> Buffer<T> {
    with_selection(bits, kind, |selection| {
        if in_place {
            let mut values = values.to_vec();
            let new_len = filter_slice_in_place(&mut values, selection);
            values.truncate(new_len);
            Buffer::copy_from(values)
        } else {
            filter_slice(values, selection, BufferAllocatorRef::static_ref())
        }
    })
}

/// Append the values of `values` that `bits` keeps, in the representation `kind`, after the
/// first value of `values`.
fn append<T: Copy>(values: &[T], bits: &BitBuffer, kind: Kind) -> Vec<T> {
    let mut out = BufferMut::from_iter(values.iter().take(1).copied());
    with_selection(bits, kind, |selection| {
        append_filtered(values, selection, &mut out)
    });
    out.as_slice().to_vec()
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

    // The branchless copy stores one value past the last selected value.
    let mut out = Vec::with_capacity(values.len() + 1);
    // SAFETY: `bits` covers `values`, and the output has room for every value plus one.
    let written = unsafe { scalar::compact_branchless(values, bits, out.spare_capacity_mut()) };
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

        let appended = append(&values, bits, kind);
        assert_eq!(appended[..1], values[..1], "{kind:?}, append prefix");
        assert_eq!(appended[1..], expected, "{kind:?}, append");
    }

    check_bitmap_strategies(&values, bits);
}

/// Filters a random selection of the given density for values of every width.
fn check_random(density: f64, offset: usize, len: usize) {
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

/// Filters runs of selected values for values of every width.
fn check_runs(run_len: usize, gap: usize, offset: usize) {
    let bits = make_bits(2000, offset, |i| i % (run_len + gap) < run_len);

    check::<u8>(&bits);
    check::<u16>(&bits);
    check::<u32>(&bits);
    check::<u64>(&bits);
    check::<u128>(&bits);
}

#[rstest]
#[cfg_attr(miri, ignore = "too slow under Miri, which runs a subset")]
fn filter_matches_expected(
    #[values(0.001, 0.05, 0.2, 0.5, 0.9, 0.999)] density: f64,
    #[values(0, 3, 64)] offset: usize,
    #[values(9, 151, 2000)] len: usize,
) {
    check_random(density, offset, len);
}

#[rstest]
#[cfg_attr(miri, ignore = "too slow under Miri, which runs a subset")]
fn filter_runs_matches_expected(
    #[values(1, 12, 200)] run_len: usize,
    #[values(1, 16, 300)] gap: usize,
    #[values(0, 5)] offset: usize,
) {
    check_runs(run_len, gap, offset);
}

/// A subset of [`filter_matches_expected`] small enough for Miri that still reaches each row of
/// the strategy ladder, in place and out of place, at aligned and unaligned offsets.
#[rstest]
// Mostly empty words.
#[case(0.001, 3, 2000)]
// Below every in-place and branchless threshold, and a length that ends mid-word.
#[case(0.2, 64, 151)]
// A selection shorter than one word.
#[case(0.5, 0, 9)]
// Dense enough to byte compress values of every width, and to compact in place.
#[case(0.9, 3, 2000)]
fn filter_strategies(#[case] density: f64, #[case] offset: usize, #[case] len: usize) {
    check_random(density, offset, len);
}

/// A subset of [`filter_runs_matches_expected`] small enough for Miri.
#[rstest]
// Alternating values, too short to copy as runs.
#[case(1, 1, 0)]
// Runs long enough in bytes to copy as runs only for values of 8 or more bytes.
#[case(12, 16, 5)]
// Long runs between long gaps.
#[case(200, 300, 5)]
fn filter_run_strategies(#[case] run_len: usize, #[case] gap: usize, #[case] offset: usize) {
    check_runs(run_len, gap, offset);
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

/// The set-word walks must visit the words of [`bits::for_each_mask_word`] that have a set bit,
/// across long empty stretches, block edges, and unaligned bitmaps.
#[rstest]
fn set_word_walks_match_full_walk(
    #[values(0, 5, 61)] offset: usize,
    #[values(1, 9, 64, 700)] gap_words: usize,
) {
    let period = 64 * (gap_words + 1);
    let bits = make_bits(5000, offset, |i| {
        i % period < 70 || i % 97 == 0 && i % 3 == 0
    });
    let bits = MaskBits::Bitmap(&bits);

    let mut expected = Vec::new();
    bits::for_each_mask_word(bits, |word, start, len| {
        if word != 0 {
            expected.push((word, start, len));
        }
    });

    let mut visited = Vec::new();
    bits::for_each_set_mask_word(bits, |word, start, len| visited.push((word, start, len)));
    assert_eq!(visited, expected);

    let mut words = Vec::new();
    assert!(bits::all_set_mask_words(bits, |word| {
        words.push(word);
        true
    }));
    assert_eq!(words.len(), expected.len());
    assert!(words.iter().all(|&word| word != 0));
}

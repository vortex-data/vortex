// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filtering of records whose byte width is known only at run time. See [`filter_records`].

use super::IN_PLACE_MIN_DENSITY;
use super::Repr;
use super::Selection;
use super::bits::MaskBits;
use super::bits::for_each_mask_word;
use super::filter_buffer;
use crate::Buffer;
use crate::BufferAllocatorRef;
use crate::ByteBuffer;
use crate::ByteBufferMut;
use crate::match_each_power_of_two_width;

/// Filters the records of `width` bytes in `values` by `selection` and returns the selected
/// records.
///
/// Power-of-two widths up to 32 bytes use [`filter_buffer`]. Other widths copy each run of
/// selected records with one copy. A uniquely owned buffer with density of at least 0.50 is
/// compacted in place, and other buffers are copied to a new allocation from `allocator`. The
/// result keeps the alignment of `values`.
///
/// # Panics
///
/// Panics if `values` does not hold one record of `width` bytes for each position of
/// `selection`.
pub fn filter_records(
    values: ByteBuffer,
    width: usize,
    selection: Selection<'_>,
    allocator: &BufferAllocatorRef,
) -> ByteBuffer {
    assert_eq!(
        values.len(),
        selection.len() * width,
        "Selection length must equal the number of records"
    );

    let alignment = values.alignment();

    match_each_power_of_two_width!(
        width,
        |W| {
            let records = Buffer::<[u8; W]>::from_byte_buffer(values);
            filter_buffer(records, selection, allocator)
                .into_byte_buffer()
                .aligned(alignment)
        },
        _ => { filter_runtime_width(values, width, selection, allocator).aligned(alignment) }
    )
}

fn filter_runtime_width(
    values: ByteBuffer,
    width: usize,
    selection: Selection<'_>,
    allocator: &BufferAllocatorRef,
) -> ByteBuffer {
    let values = if selection.density() >= IN_PLACE_MIN_DENSITY {
        match values.try_into_mut() {
            Ok(mut values_mut) => {
                let new_len = compact_in_place(values_mut.as_mut_slice(), width, selection);
                values_mut.truncate(new_len);
                return values_mut.freeze();
            }
            Err(values) => values,
        }
    } else {
        values
    };

    let mut filtered =
        ByteBufferMut::with_capacity_in(selection.true_count() * width, allocator.clone());
    for_each_run(selection, |start, len| {
        filtered.extend_from_slice(&values[start * width..(start + len) * width]);
    });

    debug_assert_eq!(filtered.len(), selection.true_count() * width);
    filtered.freeze()
}

/// Compacts the records of `width` bytes that `selection` keeps to the front of `values`, and
/// returns the new length in bytes.
fn compact_in_place(values: &mut [u8], width: usize, selection: Selection<'_>) -> usize {
    let mut write_pos = 0;

    for_each_run(selection, |start, len| {
        let source = start * width;
        let run_bytes = len * width;
        if write_pos != source {
            values.copy_within(source..source + run_bytes, write_pos);
        }
        write_pos += run_bytes;
    });

    debug_assert_eq!(write_pos, selection.true_count() * width);
    write_pos
}

/// Invokes `f` with the `(start, len)` of each run of positions that `selection` keeps, in
/// order.
///
/// A run of set bits that crosses a word boundary of a bitmap is split at the boundary.
fn for_each_run(selection: Selection<'_>, mut f: impl FnMut(usize, usize)) {
    match selection.repr {
        Repr::Slices(slices) => {
            for &(start, end) in slices {
                f(start, end - start);
            }
        }
        Repr::Indices(indices) => {
            for &index in indices {
                f(index, 1);
            }
        }
        Repr::Bitmap(bits) => {
            for_each_mask_word(MaskBits::Bitmap(bits), |mut word, word_start, _| {
                while word != 0 {
                    let run_start = word.trailing_zeros() as usize;
                    let run_len = (!(word >> run_start)).trailing_zeros() as usize;
                    f(word_start + run_start, run_len);

                    // Adding the lowest bit of the run carries through the run and clears it.
                    word &= word.wrapping_add(1 << run_start);
                }
            });
        }
    }
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::BitBuffer;

    /// A selection of `len` positions, stored at bit `offset` of its bitmap.
    fn make_bits(len: usize, offset: usize, pattern: impl Fn(usize) -> bool) -> BitBuffer {
        let backing =
            BitBuffer::from_iter(std::iter::repeat_n(false, offset).chain((0..len).map(pattern)));
        BitBuffer::new_with_offset(backing.inner().clone(), len, offset)
    }

    fn records(len: usize, width: usize) -> Vec<u8> {
        (0..len * width).map(|i| (i % 251) as u8).collect()
    }

    fn expected(values: &[u8], width: usize, bits: &BitBuffer) -> Vec<u8> {
        values
            .chunks(width)
            .zip(bits.iter())
            .filter_map(|(record, selected)| selected.then_some(record))
            .flatten()
            .copied()
            .collect()
    }

    #[rstest]
    fn filter_records_matches_expected(
        #[values(3, 4, 12, 24, 48)] width: usize,
        #[values(0, 5)] offset: usize,
        #[values(
            |i: usize| i.is_multiple_of(3),
            |i: usize| !i.is_multiple_of(16),
            |i: usize| (i / 70).is_multiple_of(2),
        )]
        pattern: fn(usize) -> bool,
        #[values(false, true)] shared: bool,
    ) {
        const LEN: usize = 151;

        let bits = make_bits(LEN, offset, pattern);
        let slices = bits.set_slices().collect::<Vec<_>>();
        let indices = bits.set_indices().collect::<Vec<_>>();
        let values = records(LEN, width);
        let expected = expected(&values, width, &bits);

        // SAFETY: all three representations come from `bits`.
        let selections = unsafe {
            [
                Selection::from_bitmap_unchecked(&bits, bits.true_count()),
                Selection::from_slices_unchecked(LEN, &slices, bits.true_count()),
                Selection::from_indices_unchecked(LEN, &indices),
            ]
        };

        for selection in selections {
            let buffer = ByteBuffer::copy_from(values.as_slice());
            let _retained = shared.then(|| buffer.clone());
            let filtered =
                filter_records(buffer, width, selection, BufferAllocatorRef::static_ref());
            assert_eq!(filtered.as_slice(), expected.as_slice());
        }
    }

    #[test]
    fn filter_records_in_place() {
        let bits = make_bits(8, 0, |i| i % 4 != 0);
        // SAFETY: the true count comes from `bits`.
        let selection = unsafe { Selection::from_bitmap_unchecked(&bits, bits.true_count()) };

        let values = ByteBuffer::copy_from(records(8, 3));
        let original_ptr = values.as_ptr();
        let filtered = filter_records(values, 3, selection, BufferAllocatorRef::static_ref());
        assert_eq!(filtered.as_ptr(), original_ptr);
        assert_eq!(filtered.as_slice(), expected(&records(8, 3), 3, &bits));
    }

    #[test]
    fn filter_zero_width_records() {
        let bits = make_bits(10, 0, |i| i % 2 == 0);
        // SAFETY: the true count comes from `bits`.
        let selection = unsafe { Selection::from_bitmap_unchecked(&bits, bits.true_count()) };

        let filtered = filter_records(
            ByteBuffer::empty(),
            0,
            selection,
            BufferAllocatorRef::static_ref(),
        );
        assert!(filtered.is_empty());
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared gather kernel for fixed-width [`Interleave`] values.
//!
//! Primitive and decimal execution turn each value into a [`Source`] and call [`gather`]. They
//! differ only in how they read a constant payload or a decoded buffer, and in the array that they
//! build from the gathered payloads.

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::vortex_panic;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use crate::array::Array;
use crate::arrays::Primitive;
use crate::match_each_unsigned_integer_ptype;

/// Physical storage for one fixed-width value of an [`Interleave`] array.
///
/// A constant keeps one payload instead of one payload for each row. `row_mask` maps each row to
/// its payload, so one gather loop reads constants and decoded buffers without a branch.
pub(super) struct Source<T> {
    /// One payload for each row, or a single payload for every row of a constant.
    data: Buffer<T>,
    /// The number of logical rows. A constant can have more rows than `data` has payloads.
    len: usize,
    /// `usize::MAX` for a decoded buffer and zero for a constant, so that `row & row_mask` is the
    /// position of the payload of `row` in `data`.
    row_mask: usize,
}

impl<T: Copy> Source<T> {
    /// Returns a source of `len` rows that all hold `payload`.
    pub(super) fn constant(payload: T, len: usize, allocator: &BufferAllocatorRef) -> Self {
        Self {
            data: Buffer::full_in(payload, 1, allocator.clone()),
            len,
            row_mask: 0,
        }
    }

    /// Returns a source with one row for each payload in `data`.
    pub(super) fn decoded(data: Buffer<T>) -> Self {
        Self {
            len: data.len(),
            data,
            row_mask: usize::MAX,
        }
    }
}

/// Gathers the payload of each output row of `array` from `sources`.
///
/// `sources` holds one [`Source`] for each value of `array`, in order. Both selectors of `array`
/// **must** be [`Primitive`] arrays, which the kernels ensure with `require_child!`.
///
/// # Panics
///
/// Panics if the selectors have different lengths, or if a selector names a value or a row that
/// does not exist.
pub(super) fn gather<T: Copy>(
    array: &Array<Interleave>,
    sources: &[Source<T>],
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    let branches = array.array_indices().as_::<Primitive>();
    let rows = array.row_indices().as_::<Primitive>();

    match_each_unsigned_integer_ptype!(branches.ptype(), |A| {
        match_each_unsigned_integer_ptype!(rows.ptype(), |R| {
            gather_rows(
                sources,
                branches.as_slice::<A>(),
                rows.as_slice::<R>(),
                allocator,
            )
        })
    })
}

fn gather_rows<T, A, R>(
    sources: &[Source<T>],
    branches: &[A],
    rows: &[R],
    allocator: &BufferAllocatorRef,
) -> Buffer<T>
where
    T: Copy,
    A: AsPrimitive<usize>,
    R: AsPrimitive<usize>,
{
    // `zip` truncates to the shorter input.
    assert_eq!(
        rows.len(),
        branches.len(),
        "interleave row_indices length does not match array_indices length",
    );

    // Write rows through a `&mut` slice. Yielding `Result<T, _>` per row splits each `[u8; W]` copy
    // into narrow moves (the `Ok` payload is misaligned), and pushing through `BufferMut` reloads
    // its length on every row.
    let len = branches.len();
    let mut output = BufferMut::with_capacity_in(len, allocator.clone());
    let slots = &mut output.spare_capacity_mut()[..len];
    for ((slot, branch), row) in slots.iter_mut().zip(branches).zip(rows) {
        let branch = (*branch).as_();
        let Some(source) = sources.get(branch) else {
            vortex_panic!(
                "interleave array index {branch} out of bounds for {} values",
                sources.len()
            );
        };
        // NB: `row & row_mask` is always in bounds for a constant, so this check is the only one
        // that rejects a row past the end of a constant.
        let row = (*row).as_();
        assert!(
            row < source.len,
            "interleave row index {row} out of bounds for length {}",
            source.len
        );
        slot.write(source.data[row & source.row_mask]);
    }
    // SAFETY: the loop panics on a bad selector, so reaching here means it wrote all `len` slots:
    // `slots`, `branches`, and `rows` all have length `len`.
    unsafe { output.set_len(len) };
    output.freeze()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use super::Source;
    use super::gather_rows;
    use crate::memory::BufferAllocatorRef;

    /// Gathers row `row` of value `branch`. Value 0 is a one-row constant, and value 1 is a one-row
    /// decoded buffer.
    fn gather_one(branch: u8, row: u8) {
        let allocator = BufferAllocatorRef::statically_allocated();
        let sources = [
            Source::constant(1u32, 1, &allocator),
            Source::decoded(buffer![2u32]),
        ];

        drop(gather_rows(&sources, &[branch], &[row], &allocator));
    }

    #[test]
    #[should_panic(expected = "interleave array index 2 out of bounds for 2 values")]
    fn panics_on_out_of_bounds_array_index() {
        gather_one(2, 0);
    }

    #[rstest]
    #[case::constant(0)]
    #[case::decoded(1)]
    #[should_panic(expected = "interleave row index 1 out of bounds for length 1")]
    fn panics_on_out_of_bounds_row_index(#[case] branch: u8) {
        gather_one(branch, 1);
    }
}

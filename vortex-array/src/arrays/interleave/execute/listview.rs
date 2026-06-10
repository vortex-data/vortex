// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! List-value execution: the optimized [`Interleave`](super::super::Interleave) path for
//! [`ListView`] values.
//!
//! A [`ListView`] addresses each list by an `(offset, size)` pair into a shared `elements` array,
//! so the gather is metadata-only: the values' `elements` arrays are concatenated (without
//! rewriting them) and each selected `(offset, size)` pair is copied into place with its offset
//! shifted past the elements of the preceding values. List elements are never copied.

use num_traits::AsPrimitive;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use super::check_selector_bounds;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::Primitive;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_unsigned_integer_ptype;
use crate::require_child;
use crate::validity::Validity;

/// Gathers `N` list values under unsigned `array_indices` / `row_indices` selectors, scattering
/// each selected `(offset, size)` pair (and its list-level validity) into the output position it
/// routes to.
pub(super) fn execute(
    array: Array<Interleave>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();

    // Drive every value and both selectors to canonical encodings so we can operate on raw
    // offsets and sizes.
    let mut array = array;
    for i in 0..num_values {
        array = require_child!(array, array.value(i), i => ListView);
    }
    array = require_child!(array, array.array_indices(), num_values => Primitive);
    array = require_child!(array, array.row_indices(), num_values + 1 => Primitive);

    let dtype = array.as_ref().dtype().clone();
    let len = array.as_ref().len();
    let nullable = dtype.is_nullable();

    // The interleave's dtype already carries the recursive nullability union of the values'
    // element dtypes; casting every value's elements to it makes all chunks of the concatenated
    // elements array identical.
    let elements_dtype = match &dtype {
        DType::List(elements_dtype, _) => elements_dtype.as_ref().clone(),
        other => vortex_panic!("listview interleave requires a list dtype, got {other}"),
    };

    // Concatenate the elements without copying, recording for each value the shift its offsets
    // need to address its own slice of the concatenation.
    let mut chunks = Vec::with_capacity(num_values);
    let mut value_shifts = Vec::with_capacity(num_values);
    let mut value_offsets = Vec::with_capacity(num_values);
    let mut value_sizes = Vec::with_capacity(num_values);
    let mut value_validity = Vec::with_capacity(num_values);
    let mut shift = 0u64;
    for i in 0..num_values {
        let value = array.value(i).as_::<ListView>();
        let elements = value.elements().cast(elements_dtype.clone())?;
        value_shifts.push(shift);
        shift += elements.len() as u64;
        push_element_chunks(elements, &mut chunks);
        value_offsets.push(to_u64(value.offsets(), ctx)?);
        value_sizes.push(to_u64(value.sizes(), ctx)?);
        let validity = nullable
            .then(|| value.validity()?.execute_mask(value.len(), ctx))
            .transpose()?;
        value_validity.push(validity);
    }
    let elements = ChunkedArray::try_new(chunks, elements_dtype)?.into_array();

    let array_indices = array.array_indices().as_::<Primitive>();
    let row_indices = array.row_indices().as_::<Primitive>();
    let (offsets, sizes, validity) =
        match_each_unsigned_integer_ptype!(array_indices.ptype(), |A| {
            match_each_unsigned_integer_ptype!(row_indices.ptype(), |R| {
                gather(
                    len,
                    &value_shifts,
                    &value_offsets,
                    &value_sizes,
                    &value_validity,
                    array_indices.as_slice::<A>(),
                    row_indices.as_slice::<R>(),
                    nullable,
                )?
            })
        });

    let validity = match validity {
        Some(bits) => Validity::from(bits.freeze()),
        None => Validity::NonNullable,
    };
    Ok(ExecutionResult::done(ListViewArray::try_new(
        elements,
        offsets.freeze().into_array(),
        sizes.freeze().into_array(),
        validity,
    )?))
}

/// The scatter, monomorphized on the selector integer widths.
#[allow(clippy::too_many_arguments)]
fn gather<A: AsPrimitive<usize>, R: AsPrimitive<usize>>(
    len: usize,
    value_shifts: &[u64],
    value_offsets: &[Buffer<u64>],
    value_sizes: &[Buffer<u64>],
    value_validity: &[Option<Mask>],
    branches: &[A],
    rows: &[R],
    nullable: bool,
) -> VortexResult<(BufferMut<u64>, BufferMut<u64>, Option<BitBufferMut>)> {
    let value_lens: Vec<usize> = value_offsets.iter().map(|o| o.len()).collect();
    check_selector_bounds(branches, rows, &value_lens)?;

    let mut offsets = BufferMut::<u64>::with_capacity(len);
    let mut sizes = BufferMut::<u64>::with_capacity(len);
    for i in 0..len {
        let branch = branches[i].as_();
        let row = rows[i].as_();
        offsets.push(value_offsets[branch][row] + value_shifts[branch]);
        sizes.push(value_sizes[branch][row]);
    }

    // A missing per-value mask means every row of that value is valid; only materialized when the
    // output can be null.
    let validity = nullable.then(|| {
        BitBufferMut::collect_bool(len, |i| {
            value_validity[branches[i].as_()]
                .as_ref()
                .is_none_or(|mask| mask.value(rows[i].as_()))
        })
    });

    Ok((offsets, sizes, validity))
}

/// Appends `array`'s element chunks to `chunks`, flattening a top-level [`ChunkedArray`] so the
/// concatenated elements never nest chunked arrays.
fn push_element_chunks(array: ArrayRef, chunks: &mut Vec<ArrayRef>) {
    match array.as_opt::<Chunked>() {
        Some(chunked) => chunks.extend(chunked.iter_chunks().cloned()),
        None => chunks.push(array),
    }
}

/// Read a non-nullable integer array into a `u64` buffer.
fn to_u64(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Buffer<u64>> {
    array
        .clone()
        .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
        .execute::<Buffer<u64>>(ctx)
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::InterleaveArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::validity::Validity;

    fn selectors(indices: &[(u32, u32)]) -> (ArrayRef, ArrayRef) {
        (
            PrimitiveArray::from_iter(indices.iter().map(|&(a, _)| a)).into_array(),
            PrimitiveArray::from_iter(indices.iter().map(|&(_, r)| r)).into_array(),
        )
    }

    fn list_view(
        elements: ArrayRef,
        offsets: ArrayRef,
        sizes: ArrayRef,
        validity: Validity,
    ) -> ArrayRef {
        ListViewArray::try_new(elements, offsets, sizes, validity)
            .vortex_expect("test list view construction")
            .into_array()
    }

    #[test]
    fn interleave_lists_reorders_and_repeats() -> VortexResult<()> {
        // [[1, 2], [3], [4, 5, 6]]
        let v0 = list_view(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 2, 3].into_array(),
            buffer![2u32, 1, 3].into_array(),
            Validity::NonNullable,
        );
        // [[10], [20, 21]] expressed with out-of-order offsets.
        let v1 = list_view(
            buffer![20i32, 21, 10].into_array(),
            buffer![2u32, 0].into_array(),
            buffer![1u32, 2].into_array(),
            Validity::NonNullable,
        );
        let (array_indices, row_indices) = selectors(&[(1, 1), (0, 2), (0, 0), (1, 0), (0, 0)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;

        // [[20, 21], [4, 5, 6], [1, 2], [10], [1, 2]]
        let expected = list_view(
            buffer![20i32, 21, 4, 5, 6, 1, 2, 10, 1, 2].into_array(),
            buffer![0u32, 2, 5, 7, 8].into_array(),
            buffer![2u32, 3, 2, 1, 2].into_array(),
            Validity::NonNullable,
        );
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }

    #[test]
    fn interleave_lists_with_null_lists() -> VortexResult<()> {
        // [[1], null]
        let v0 = list_view(
            buffer![1i32].into_array(),
            buffer![0u32, 1].into_array(),
            buffer![1u32, 0].into_array(),
            Validity::Array(BoolArray::from_iter([true, false]).into_array()),
        );
        // [null, [2, 3]]
        let v1 = list_view(
            buffer![2i32, 3].into_array(),
            buffer![0u32, 0].into_array(),
            buffer![0u32, 2].into_array(),
            Validity::Array(BoolArray::from_iter([false, true]).into_array()),
        );
        let (array_indices, row_indices) = selectors(&[(1, 1), (0, 1), (1, 0), (0, 0)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;

        // [[2, 3], null, null, [1]]
        let expected = list_view(
            buffer![2i32, 3, 1].into_array(),
            buffer![0u32, 2, 2, 2].into_array(),
            buffer![2u32, 0, 0, 1].into_array(),
            Validity::Array(BoolArray::from_iter([true, false, false, true]).into_array()),
        );
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }

    #[test]
    fn interleave_lists_unifies_element_nullability() -> VortexResult<()> {
        // Non-nullable elements in v0, nullable elements (with a null) in v1: the interleave's
        // element dtype is the recursive nullability union, and v0's elements are cast to it
        // before concatenation.
        let v0 = list_view(
            buffer![1i32, 2].into_array(),
            buffer![0u32].into_array(),
            buffer![2u32].into_array(),
            Validity::NonNullable,
        );
        let v1 = list_view(
            PrimitiveArray::from_option_iter([Some(3i32), None]).into_array(),
            buffer![0u32].into_array(),
            buffer![2u32].into_array(),
            Validity::NonNullable,
        );
        let (array_indices, row_indices) = selectors(&[(1, 0), (0, 0), (1, 0)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;

        // [[3, null], [1, 2], [3, null]]
        let expected = list_view(
            PrimitiveArray::from_option_iter([Some(3i32), None, Some(1), Some(2)]).into_array(),
            buffer![0u32, 2, 0].into_array(),
            buffer![2u32, 2, 2].into_array(),
            Validity::NonNullable,
        );
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }
}

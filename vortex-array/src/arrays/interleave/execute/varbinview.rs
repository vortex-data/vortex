// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Utf8/binary-value execution: the optimized [`Interleave`](super::super::Interleave) path for
//! [`VarBinView`] values.
//!
//! The gather is metadata-only: the values' data buffers are pooled (deduplicated) into one buffer
//! set, and each selected 16-byte view is copied into place with its buffer index remapped into
//! the pool by a branchless mask blend (see [`gather`]). String bytes are never copied.

use num_traits::AsPrimitive;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use super::check_selector_bounds;
use crate::array::Array;
use crate::arrays::Primitive;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::builders::DeduplicatedBuffers;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_unsigned_integer_ptype;
use crate::require_child;
use crate::validity::Validity;

/// Gathers `N` utf8/binary values under unsigned `array_indices` / `row_indices` selectors,
/// scattering each selected view (and its validity) into the output position it routes to.
pub(super) fn execute(
    array: Array<Interleave>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();

    // Drive every value and both selectors to canonical encodings so we can operate on raw views.
    let mut array = array;
    for i in 0..num_values {
        array = require_child!(array, array.value(i), i => VarBinView);
    }
    array = require_child!(array, array.array_indices(), num_values => Primitive);
    array = require_child!(array, array.row_indices(), num_values + 1 => Primitive);

    let dtype = array.as_ref().dtype().clone();
    let len = array.as_ref().len();
    let nullable = dtype.is_nullable();

    // Pool every value's data buffers, remembering each value's mapping from its original buffer
    // indices into the pool.
    let mut buffers = DeduplicatedBuffers::default();
    let mut value_views = Vec::with_capacity(num_values);
    let mut value_lookups = Vec::with_capacity(num_values);
    let mut value_validity = Vec::with_capacity(num_values);
    for i in 0..num_values {
        let value = array.value(i).as_::<VarBinView>();
        let mut lookup =
            buffers.extend_from_iter(value.data_buffers().iter().map(|b| b.as_host().clone()));
        // The per-row remap loads from the lookup unconditionally; give buffer-less values a
        // sentinel slot so the clamped load stays in bounds.
        if lookup.is_empty() {
            lookup.push(0);
        }
        value_lookups.push(lookup);
        let validity = nullable
            .then(|| value.validity()?.execute_mask(value.len(), ctx))
            .transpose()?;
        value_views.push(value.data().views());
        value_validity.push(validity);
    }

    let array_indices = array.array_indices().as_::<Primitive>();
    let row_indices = array.row_indices().as_::<Primitive>();
    let (views, validity) = match_each_unsigned_integer_ptype!(array_indices.ptype(), |A| {
        match_each_unsigned_integer_ptype!(row_indices.ptype(), |R| {
            gather(
                len,
                &value_views,
                &value_lookups,
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
    // SAFETY: every gathered view comes from a validated value array; outlined views keep their
    // size/prefix/offset and only have their buffer index remapped into the deduplicated pool,
    // while rows routed to null inputs are written as empty views and marked null.
    let result = unsafe {
        VarBinViewArray::new_unchecked(views.freeze(), buffers.finish(), dtype, validity)
    };
    Ok(ExecutionResult::done(result))
}

/// The scatter, monomorphized on the selector integer widths.
///
/// The per-row loop is branchless: each 16-byte view is loaded as a `u128`, the remapped buffer
/// index is loaded unconditionally through a clamped lookup, and bit masks select between the
/// original bits (inlined views, `len <= 12`), the remapped index bits (outlined views), and an
/// all-zero empty view (null rows).
#[allow(clippy::too_many_arguments)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the casts deliberately extract 32-bit fields of the 128-bit view representation"
)]
fn gather<A: AsPrimitive<usize>, R: AsPrimitive<usize>>(
    len: usize,
    value_views: &[&[BinaryView]],
    value_lookups: &[Vec<u32>],
    value_validity: &[Option<Mask>],
    branches: &[A],
    rows: &[R],
    nullable: bool,
) -> VortexResult<(BufferMut<BinaryView>, Option<BitBufferMut>)> {
    /// Bits 64..96 of a raw view: the buffer index of an outlined view.
    const BUFFER_INDEX_BITS: u128 = (u32::MAX as u128) << 64;

    let value_lens: Vec<usize> = value_views.iter().map(|v| v.len()).collect();
    check_selector_bounds(branches, rows, &value_lens)?;

    // A missing per-value mask means every row of that value is valid.
    let is_valid = |i: usize| {
        value_validity[branches[i].as_()]
            .as_ref()
            .is_none_or(|mask| mask.value(rows[i].as_()))
    };

    let mut views = BufferMut::<BinaryView>::with_capacity(len);
    for i in 0..len {
        let branch = branches[i].as_();
        let raw = value_views[branch][rows[i].as_()].as_u128();
        let lookup = &value_lookups[branch];

        // Load the remapped buffer index unconditionally: an inlined view carries payload bytes in
        // the index bits and a null row's view may carry arbitrary bits, so the index is clamped
        // into the (never empty) lookup table and the loaded value discarded by the masks below.
        let buffer_index = (raw >> 64) as u32 as usize;
        let remapped = u128::from(lookup[buffer_index.min(lookup.len() - 1)]) << 64;

        // All-ones when the view is outlined (`len > 12`, read from the view's size bits), zero
        // when inlined.
        let outlined =
            u128::from((raw as u32) > BinaryView::MAX_INLINED_SIZE as u32).wrapping_neg();
        // All-ones when the row is valid. A null row's view bits are arbitrary, so they are zeroed
        // into an empty view rather than left masquerading as a (dangling) outlined view.
        let valid = u128::from(is_valid(i)).wrapping_neg();

        let blended = (raw & !(BUFFER_INDEX_BITS & outlined)) | (remapped & outlined);
        views.push(BinaryView::from(blended & valid));
    }

    let validity = nullable.then(|| BitBufferMut::collect_bool(len, is_valid));

    Ok((views, validity))
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::InterleaveArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;

    fn selectors(indices: &[(u32, u32)]) -> (ArrayRef, ArrayRef) {
        (
            PrimitiveArray::from_iter(indices.iter().map(|&(a, _)| a)).into_array(),
            PrimitiveArray::from_iter(indices.iter().map(|&(_, r)| r)).into_array(),
        )
    }

    #[test]
    fn interleave_strings_inlined_and_outlined() -> VortexResult<()> {
        let v0 = VarBinViewArray::from_iter_str([
            "short",
            "an outlined string longer than twelve bytes",
            "tiny",
        ])
        .into_array();
        let v1 = VarBinViewArray::from_iter_str(["another outlined string, also long", "in"])
            .into_array();
        let (array_indices, row_indices) =
            selectors(&[(1, 0), (0, 2), (0, 1), (1, 1), (0, 1), (0, 0)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;

        let expected = VarBinViewArray::from_iter_str([
            "another outlined string, also long",
            "tiny",
            "an outlined string longer than twelve bytes",
            "in",
            "an outlined string longer than twelve bytes",
            "short",
        ]);
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }

    #[test]
    fn interleave_all_inlined_with_outlined() -> VortexResult<()> {
        // v0 is entirely inlined and may contribute no data buffers, exercising the sentinel
        // lookup slot in the branchless remap.
        let v0 = VarBinViewArray::from_iter_str(["a", "bb"]).into_array();
        let v1 = VarBinViewArray::from_iter_str(["an outlined string longer than twelve bytes"])
            .into_array();
        let (array_indices, row_indices) = selectors(&[(0, 1), (1, 0), (0, 0)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;

        let expected = VarBinViewArray::from_iter_str([
            "bb",
            "an outlined string longer than twelve bytes",
            "a",
        ]);
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }

    #[test]
    fn interleave_nullable_strings() -> VortexResult<()> {
        let v0 = VarBinViewArray::from_iter_nullable_str([
            Some("a string that is too long to inline"),
            None,
        ])
        .into_array();
        let v1 = VarBinViewArray::from_iter_nullable_str([None, Some("ok")]).into_array();
        let (array_indices, row_indices) = selectors(&[(0, 1), (1, 1), (0, 0), (1, 0)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;

        let expected = VarBinViewArray::from_iter_nullable_str([
            None,
            Some("ok"),
            Some("a string that is too long to inline"),
            None,
        ]);
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }

    #[test]
    fn interleave_binary_values() -> VortexResult<()> {
        let v0 = VarBinViewArray::from_iter_bin([
            b"binary value that exceeds the inline limit".as_slice(),
            b"\x00\x01".as_slice(),
        ])
        .into_array();
        let v1 = VarBinViewArray::from_iter_bin([b"abc".as_slice()]).into_array();
        let (array_indices, row_indices) = selectors(&[(0, 1), (1, 0), (0, 0)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;

        let expected = VarBinViewArray::from_iter_bin([
            b"\x00\x01".as_slice(),
            b"abc".as_slice(),
            b"binary value that exceeds the inline limit".as_slice(),
        ]);
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Patched;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::PatchedView;
use crate::arrays::primitive::PrimitiveDataParts;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;

impl TakeExecute for Patched {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only pushdown take when we have primitive types.
        if !array.dtype().is_primitive() {
            return Ok(None);
        }

        // Perform take on the inner array, including the placeholders.
        let inner = array
            .inner()
            .take(indices.clone())?
            .execute::<PrimitiveArray>(ctx)?;

        let PrimitiveDataParts {
            buffer,
            validity,
            ptype,
        } = inner.into_data_parts();

        let take_indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let patch_indices = array
            .patch_indices()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let patch_values = array
            .patch_values()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let chunk_offsets = array
            .chunk_offsets()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let view = PatchedView::new(
            array.offset(),
            array.len(),
            patch_indices.as_slice::<u16>(),
            chunk_offsets.as_slice::<u32>(),
        );

        match_each_unsigned_integer_ptype!(take_indices.ptype(), |I| {
            match_each_native_ptype!(ptype, |V| {
                let mut output = Buffer::<V>::from_byte_buffer(buffer.unwrap_host()).into_mut();
                take_patches(
                    output.as_mut(),
                    take_indices.as_slice::<I>(),
                    view,
                    patch_values.as_slice::<V>(),
                );

                // SAFETY: output and validity still have same length after take_patches returns.
                unsafe {
                    Ok(Some(
                        PrimitiveArray::new_unchecked(output.freeze(), validity).into_array(),
                    ))
                }
            })
        })
    }
}

/// Overwrite each taken row that lands on a patch with the patch value.
///
/// Every lookup is a constant-time chunk select plus a binary search over at most one chunk of
/// indices. Null take indices may hold any value; out-of-range rows simply find no patch.
fn take_patches<I: IntegerPType, V: NativePType>(
    output: &mut [V],
    take_indices: &[I],
    view: PatchedView<'_>,
    patch_values: &[V],
) {
    for (output_index, index) in take_indices.iter().enumerate() {
        if let Some(ordinal) = view.search(index.as_()).to_found() {
            output[output_index] = patch_values[ordinal];
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::Patched;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::patches::Patches;
    use crate::validity::Validity;

    fn make_patched_array(
        base: &[u16],
        patch_indices: &[u32],
        patch_values: &[u16],
        slice: Range<usize>,
    ) -> VortexResult<ArrayRef> {
        let values = PrimitiveArray::from_iter(base.iter().copied()).into_array();
        let patches = Patches::new(
            base.len(),
            0,
            PrimitiveArray::from_iter(patch_indices.iter().copied()).into_array(),
            PrimitiveArray::from_iter(patch_values.iter().copied()).into_array(),
            None,
        )?;

        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();

        Patched::from_array_and_patches(values, &patches, &mut ctx)?
            .into_array()
            .slice(slice)
    }

    #[test]
    fn test_take_basic() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Array with base values [0, 0, 0, 0, 0] patched at indices [1, 3] with values [10, 30]
        let array = make_patched_array(&[0; 5], &[1, 3], &[10, 30], 0..5)?;

        // Take indices [0, 1, 2, 3, 4] - should get [0, 10, 0, 30, 0]
        let indices = buffer![0u32, 1, 2, 3, 4].into_array();
        #[expect(deprecated)]
        let result = array.take(indices)?.to_canonical()?.into_array();

        let expected = PrimitiveArray::from_iter([0u16, 10, 0, 30, 0]).into_array();
        assert_arrays_eq!(expected, result, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_take_sliced() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = make_patched_array(&[0; 10], &[1, 3], &[100, 200], 2..10)?;

        let indices = buffer![0u32, 1, 2, 3, 7].into_array();
        #[expect(deprecated)]
        let result = array.take(indices)?.to_canonical()?.into_array();

        let expected = PrimitiveArray::from_iter([0u16, 200, 0, 0, 0]).into_array();
        assert_arrays_eq!(expected, result, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_take_across_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let base: Vec<u16> = (0..4096).map(|i| (i % 7) as u16).collect();
        let array = make_patched_array(
            &base,
            &[5, 1030, 1031, 2200, 4095],
            &[1, 2, 3, 4, 5],
            0..4096,
        )?;

        let indices = buffer![4095u32, 1031, 6, 2200, 1030, 5, 2199].into_array();
        #[expect(deprecated)]
        let result = array.take(indices)?.to_canonical()?.into_array();

        let expected =
            PrimitiveArray::from_iter([5u16, 3, base[6], 4, 2, 1, base[2199]]).into_array();
        assert_arrays_eq!(expected, result, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_take_out_of_order() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Array with base values [0, 0, 0, 0, 0] patched at indices [1, 3] with values [10, 30]
        let array = make_patched_array(&[0; 5], &[1, 3], &[10, 30], 0..5)?;

        // Take indices in reverse order
        let indices = buffer![4u32, 3, 2, 1, 0].into_array();
        #[expect(deprecated)]
        let result = array.take(indices)?.to_canonical()?.into_array();

        let expected = PrimitiveArray::from_iter([0u16, 30, 0, 10, 0]).into_array();
        assert_arrays_eq!(expected, result, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_take_duplicates() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Array with base values [0, 0, 0, 0, 0] patched at index [2] with value [99]
        let array = make_patched_array(&[0; 5], &[2], &[99], 0..5)?;

        // Take the same patched index multiple times
        let indices = buffer![2u32, 2, 0, 2].into_array();
        #[expect(deprecated)]
        let result = array.take(indices)?.to_canonical()?.into_array();

        let expected = PrimitiveArray::from_iter([99u16, 99, 0, 99]).into_array();
        assert_arrays_eq!(expected, result, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_take_with_null_indices() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        // Array: 10 elements, base value 0, patches at indices 2, 5, 8 with values 20, 50, 80
        let array = make_patched_array(&[0; 10], &[2, 5, 8], &[20, 50, 80], 0..10)?;

        // Take 10 indices, with nulls at positions 2, 5, 8.
        let indices = PrimitiveArray::new(
            buffer![0u32, 2, 2, 5, 8, 0, 5, 8, 3, 1],
            Validity::Array(
                BoolArray::from_iter([
                    true, true, false, true, true, false, true, true, false, true,
                ])
                .into_array(),
            ),
        );
        #[expect(deprecated)]
        let result = array
            .take(indices.into_array())?
            .to_canonical()?
            .into_array();

        // Expected: [0, 20, null, 50, 80, null, 50, 80, null, 0]
        let expected = PrimitiveArray::new(
            buffer![0u16, 20, 0, 50, 80, 0, 50, 80, 0, 0],
            Validity::Array(
                BoolArray::from_iter([
                    true, true, false, true, true, false, true, true, false, true,
                ])
                .into_array(),
            ),
        );
        assert_arrays_eq!(expected.into_array(), result, &mut ctx);

        Ok(())
    }
}

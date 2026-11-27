// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::arrays::dict::TakeExecute;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;

impl TakeExecute for Patched {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let base = array.inner().take(indices.clone())?;
        // The base take already supplies nulls for null indices.
        super::select_rows(array, base, array.patches().take(indices, ctx)?, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Patched;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::patches::Patches;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn make_patched_array(
        base: &[u16],
        patch_indices: &[u32],
        patch_values: &[u16],
        slice: Range<usize>,
    ) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter(base.iter().copied()).into_array();
        let patches = Patches::new(
            base.len(),
            0,
            PrimitiveArray::from_iter(patch_indices.iter().copied()).into_array(),
            PrimitiveArray::from_iter(patch_values.iter().copied()).into_array(),
            None,
            &mut ctx,
        )?;

        let mut ctx = SESSION.create_execution_ctx();

        Patched::from_array_and_patches(values, &patches, &mut ctx)?
            .into_array()
            .slice(slice)
    }

    #[test]
    fn test_take_basic() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Array with base values [0, 0, 0, 0, 0] patched at indices [1, 3] with values [10, 30]
        let array = make_patched_array(&[0; 5], &[1, 3], &[10, 30], 0..5)?;

        // Take indices [0, 1, 2, 3, 4] - should get [0, 10, 0, 30, 0]
        let indices = buffer![0u32, 1, 2, 3, 4].into_array();
        let result = array.take(indices)?;

        let expected = PrimitiveArray::from_iter([0u16, 10, 0, 30, 0]).into_array();
        assert_arrays_eq!(result, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_take_sliced() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = make_patched_array(&[0; 10], &[1, 3], &[100, 200], 2..10)?;

        let indices = buffer![0u32, 1, 2, 3, 7].into_array();
        let result = array.take(indices)?;

        let expected = PrimitiveArray::from_iter([0u16, 200, 0, 0, 0]).into_array();
        assert_arrays_eq!(result, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_take_out_of_order() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Array with base values [0, 0, 0, 0, 0] patched at indices [1, 3] with values [10, 30]
        let array = make_patched_array(&[0; 5], &[1, 3], &[10, 30], 0..5)?;

        // Take indices in reverse order
        let indices = buffer![4u32, 3, 2, 1, 0].into_array();
        let result = array.take(indices)?;

        let expected = PrimitiveArray::from_iter([0u16, 30, 0, 10, 0]).into_array();
        assert_arrays_eq!(result, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_take_duplicates() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Array with base values [0, 0, 0, 0, 0] patched at index [2] with value [99]
        let array = make_patched_array(&[0; 5], &[2], &[99], 0..5)?;

        // Take the same patched index multiple times
        let indices = buffer![2u32, 2, 0, 2].into_array();
        let result = array.take(indices)?;

        let expected = PrimitiveArray::from_iter([99u16, 99, 0, 99]).into_array();
        assert_arrays_eq!(result, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_take_with_null_indices() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        use crate::arrays::BoolArray;
        use crate::validity::Validity;

        // Array: 10 elements, base value 0, patches at indices 2, 5, 8 with values 20, 50, 80
        let array = make_patched_array(&[0; 10], &[2, 5, 8], &[20, 50, 80], 0..10)?;

        // Take 10 indices, with nulls at positions 1, 4, 7
        // Indices: [0, 2, 2, 5, 8, 0, 5, 8, 3, 1]
        // Nulls:   [ ,  , N,  ,  , N,  ,  , N,  ]
        // Position 2 (index=2, patched) is null
        // Position 5 (index=0, unpatched) is null
        // Position 8 (index=3, unpatched) is null
        let indices = PrimitiveArray::new(
            buffer![0u32, 2, 2, 5, 8, 0, 5, 8, 3, 1],
            Validity::Array(
                BoolArray::from_iter([
                    true, true, false, true, true, false, true, true, false, true,
                ])
                .into_array(),
            ),
        );
        let result = array.take(indices.into_array())?;

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
        assert_arrays_eq!(result, expected.into_array(), &mut ctx);

        Ok(())
    }
}

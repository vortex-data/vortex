// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use rustc_hash::FxHashMap;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Patched;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::patched::PatchLayout;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

impl TakeExecute for Patched {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Take the inner array, which includes the filler values, and then patch the taken rows
        // whose source was patched.
        let inner = array.inner().take(indices.clone())?;

        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let indices_valid = indices
            .as_ref()
            .validity()?
            .execute_mask(indices.len(), ctx)?;
        let layout = array.execute_layout(ctx)?;
        let layout = layout.layout()?;

        let (positions, sources) = match_each_integer_ptype!(indices.ptype(), |I| {
            take_patches(indices.as_slice::<I>(), &indices_valid, &layout)
        });

        if positions.is_empty() {
            return Ok(Some(inner));
        }

        let values = array
            .patch_values()
            .take(PrimitiveArray::new(sources.freeze(), Validity::NonNullable).into_array())?;
        let positions = PrimitiveArray::new(positions.freeze(), Validity::NonNullable).into_array();
        Ok(Some(
            Patched::try_new(inner, positions, values, ctx)?.into_array(),
        ))
    }
}

/// Returns the positions of the taken rows that are patched, in ascending order, along with the
/// index of the patch value for each.
fn take_patches<I: AsPrimitive<usize>>(
    indices: &[I],
    indices_valid: &Mask,
    layout: &PatchLayout<'_>,
) -> (BufferMut<u32>, BufferMut<u32>) {
    let mut positions = BufferMut::<u32>::empty();
    let mut sources = BufferMut::<u32>::empty();
    let n_patches = layout.all_patches().len();
    if n_patches == 0 {
        return (positions, sources);
    }

    let mut emit = |position: usize, patch: Option<usize>| {
        if let Some(patch) = patch
            && indices_valid.value(position)
        {
            #[expect(clippy::cast_possible_truncation, reason = "patch counts fit in u32")]
            {
                positions.push(position as u32);
                sources.push(patch as u32);
            }
        }
    };

    // A dense lookup table costs a pass over the array, so only use it when taking many rows.
    if layout.len() <= indices.len().saturating_mul(4) {
        let mut lookup = vec![u32::MAX; layout.len()];
        #[expect(clippy::cast_possible_truncation, reason = "patch counts fit in u32")]
        layout.for_each(|pos, patch| lookup[pos] = patch as u32);
        for (position, index) in indices.iter().enumerate() {
            // Null indices may hold any value, so avoid indexing out of bounds.
            let patch = lookup.get(index.as_()).copied().unwrap_or(u32::MAX);
            emit(position, (patch != u32::MAX).then_some(patch as usize));
        }
    } else {
        let mut lookup = FxHashMap::with_capacity_and_hasher(n_patches, Default::default());
        layout.for_each(|pos, patch| {
            lookup.insert(pos, patch);
        });
        for (position, index) in indices.iter().enumerate() {
            emit(position, lookup.get(&index.as_()).copied());
        }
    }

    (positions, sources)
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

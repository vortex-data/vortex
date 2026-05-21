// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rustc_hash::FxHashMap;
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

        let indices_ptype = indices.dtype().as_ptype();

        match_each_unsigned_integer_ptype!(indices_ptype, |I| {
            match_each_native_ptype!(ptype, |V| {
                let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
                let lane_offsets = array
                    .lane_offsets()
                    .clone()
                    .execute::<PrimitiveArray>(ctx)?;
                let patch_indices = array
                    .patch_indices()
                    .clone()
                    .execute::<PrimitiveArray>(ctx)?;
                let patch_values = array
                    .patch_values()
                    .clone()
                    .execute::<PrimitiveArray>(ctx)?;
                let mut output = Buffer::<V>::from_byte_buffer(buffer.unwrap_host()).into_mut();
                take_map(
                    output.as_mut(),
                    indices.as_slice::<I>(),
                    array.offset(),
                    array.len(),
                    array.n_lanes(),
                    lane_offsets.as_slice::<u32>(),
                    patch_indices.as_slice::<u16>(),
                    patch_values.as_slice::<V>(),
                );

                // SAFETY: output and validity still have same length after take_map returns.
                unsafe {
                    Ok(Some(
                        PrimitiveArray::new_unchecked(output.freeze(), validity).into_array(),
                    ))
                }
            })
        })
    }
}

/// Take patches for the given `indices` and apply them onto an `output` using a hash map.
///
/// First, builds a hashmap from index to patch value, then uses the hashmap in a loop to collect
/// the values.
#[expect(clippy::too_many_arguments)]
fn take_map<I: IntegerPType, V: NativePType>(
    output: &mut [V],
    indices: &[I],
    offset: usize,
    len: usize,
    n_lanes: usize,
    lane_offsets: &[u32],
    patch_index: &[u16],
    patch_value: &[V],
) {
    let n_chunks = (offset + len).div_ceil(1024);
    // Build a hashmap of patch_index -> values.
    let mut index_map = FxHashMap::with_capacity_and_hasher(patch_index.len(), Default::default());
    for chunk in 0..n_chunks {
        for lane in 0..n_lanes {
            let lane_start = lane_offsets[chunk * n_lanes + lane];
            let lane_end = lane_offsets[chunk * n_lanes + lane + 1];
            for i in lane_start..lane_end {
                let patch_idx = patch_index[i as usize];
                let patch_value = patch_value[i as usize];

                let index = chunk * 1024 + patch_idx as usize;
                if index >= offset && index < offset + len {
                    index_map.insert(index - offset, patch_value);
                }
            }
        }
    }

    // Now, iterate the take indices using the prebuilt hashmap.
    // Undefined/null indices will miss the hash map, which we can ignore.
    for (output_index, index) in indices.iter().enumerate() {
        let index = index.as_();
        if let Some(&patch_value) = index_map.get(&index) {
            output[output_index] = patch_value;
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
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::arrays::Patched;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::patches::Patches;

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
        let mut ctx = ExecutionCtx::new(session);

        Patched::from_array_and_patches(values, &patches, &mut ctx)?
            .into_array()
            .slice(slice)
    }

    #[test]
    fn test_take_basic() -> VortexResult<()> {
        // Array with base values [0, 0, 0, 0, 0] patched at indices [1, 3] with values [10, 30]
        let array = make_patched_array(&[0; 5], &[1, 3], &[10, 30], 0..5)?;

        // Take indices [0, 1, 2, 3, 4] - should get [0, 10, 0, 30, 0]
        let indices = buffer![0u32, 1, 2, 3, 4].into_array();
        #[expect(deprecated)]
        let result = array.take(indices)?.to_canonical()?.into_array();

        let expected = PrimitiveArray::from_iter([0u16, 10, 0, 30, 0]).into_array();
        assert_arrays_eq!(expected, result);

        Ok(())
    }

    #[test]
    fn test_take_sliced() -> VortexResult<()> {
        let array = make_patched_array(&[0; 10], &[1, 3], &[100, 200], 2..10)?;

        let indices = buffer![0u32, 1, 2, 3, 7].into_array();
        #[expect(deprecated)]
        let result = array.take(indices)?.to_canonical()?.into_array();

        let expected = PrimitiveArray::from_iter([0u16, 200, 0, 0, 0]).into_array();
        assert_arrays_eq!(expected, result);

        Ok(())
    }

    #[test]
    fn test_take_out_of_order() -> VortexResult<()> {
        // Array with base values [0, 0, 0, 0, 0] patched at indices [1, 3] with values [10, 30]
        let array = make_patched_array(&[0; 5], &[1, 3], &[10, 30], 0..5)?;

        // Take indices in reverse order
        let indices = buffer![4u32, 3, 2, 1, 0].into_array();
        #[expect(deprecated)]
        let result = array.take(indices)?.to_canonical()?.into_array();

        let expected = PrimitiveArray::from_iter([0u16, 30, 0, 10, 0]).into_array();
        assert_arrays_eq!(expected, result);

        Ok(())
    }

    #[test]
    fn test_take_duplicates() -> VortexResult<()> {
        // Array with base values [0, 0, 0, 0, 0] patched at index [2] with value [99]
        let array = make_patched_array(&[0; 5], &[2], &[99], 0..5)?;

        // Take the same patched index multiple times
        let indices = buffer![2u32, 2, 0, 2].into_array();
        #[expect(deprecated)]
        let result = array.take(indices)?.to_canonical()?.into_array();

        // execute the array.
        #[expect(deprecated)]
        let _canonical = result.to_canonical()?.into_primitive();

        let expected = PrimitiveArray::from_iter([99u16, 99, 0, 99]).into_array();
        assert_arrays_eq!(expected, result);

        Ok(())
    }

    #[test]
    fn test_take_with_null_indices() -> VortexResult<()> {
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
        assert_arrays_eq!(expected.into_array(), result);

        Ok(())
    }
}

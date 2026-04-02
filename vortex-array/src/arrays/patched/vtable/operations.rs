// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::Patched;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;

impl OperationsVTable<Patched> for Patched {
    fn scalar_at(
        array: ArrayView<'_, Patched>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let chunk = (index + array.offset) / 1024;

        #[expect(
            clippy::cast_possible_truncation,
            reason = "N % 1024 always fits in u16"
        )]
        let chunk_index = ((index + array.offset) % 1024) as u16;

        let lane = (index + array.offset) % array.n_lanes;

        let range = array.lane_range(chunk, lane)?;

        // Get the range of indices corresponding to the lane, potentially decoding them to avoid
        // the overhead of repeated scalar_at calls.
        let patch_indices = array
            .patch_indices()
            .slice(range.clone())?
            .optimize()?
            .execute::<PrimitiveArray>(ctx)?;

        // NOTE: we do linear scan as lane has <= 32 patches, binary search would likely
        //  be slower.
        for (&patch_index, idx) in std::iter::zip(patch_indices.as_slice::<u16>(), range) {
            if patch_index == chunk_index {
                return array.patch_values().scalar_at(idx)?.cast(array.dtype());
            }
        }

        // Otherwise, access the underlying value.
        array.base_array().scalar_at(index)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::arrays::Patched;
    use crate::arrays::PatchedArray;
    use crate::dtype::Nullability;
    use crate::optimizer::ArrayOptimizer;
    use crate::patches::Patches;
    use crate::scalar::Scalar;

    #[test]
    fn test_simple() {
        let values = buffer![0u16; 1024].into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
        )
        .unwrap();

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        let array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        assert_eq!(
            array.scalar_at(0).unwrap(),
            Scalar::primitive(0u16, Nullability::NonNullable)
        );
        assert_eq!(
            array.scalar_at(1).unwrap(),
            Scalar::primitive(1u16, Nullability::NonNullable)
        );
        assert_eq!(
            array.scalar_at(2).unwrap(),
            Scalar::primitive(1u16, Nullability::NonNullable)
        );
        assert_eq!(
            array.scalar_at(3).unwrap(),
            Scalar::primitive(1u16, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_multi_chunk() {
        let values = buffer![0u16; 4096].into_array();
        let patches = Patches::new(
            4096,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
        )
        .unwrap();

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        let array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        for index in 0..array.len() {
            let value = array.scalar_at(index).unwrap();

            if [1, 2, 3].contains(&index) {
                assert_eq!(value, 1u16.into());
            } else {
                assert_eq!(value, 0u16.into());
            }
        }
    }

    #[test]
    fn test_multi_chunk_sliced() {
        let values = buffer![0u16; 4096].into_array();
        let patches = Patches::new(
            4096,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
        )
        .unwrap();

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        let array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array()
            .slice(3..4096)
            .unwrap()
            .optimize()
            .unwrap();

        assert!(array.is::<Patched>());

        assert_eq!(array.scalar_at(0).unwrap(), 1u16.into());
        for index in 1..array.len() {
            assert_eq!(array.scalar_at(index).unwrap(), 0u16.into());
        }
    }
}

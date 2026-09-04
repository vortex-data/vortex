// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::optimizer::ArrayOptimizer;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::scalar::Scalar;

impl OperationsVTable<Patched> for Patched {
    fn scalar_at(
        array: ArrayView<'_, Patched>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let grid = array.offset() + index;
        let chunk = grid / PATCH_CHUNK_SIZE;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a position within a chunk always fits in u16"
        )]
        let local = (grid % PATCH_CHUNK_SIZE) as u16;

        let (start, stop) = chunk_bounds(array, chunk, ctx)?;
        if start == stop {
            return array.inner().execute_scalar(index, ctx);
        }

        // Decode only this chunk's indices rather than a whole compressed child.
        let chunk_indices = array
            .patch_indices()
            .slice(start..stop)?
            .optimize()?
            .execute::<PrimitiveArray>(ctx)?;

        match chunk_indices.as_slice::<u16>().binary_search(&local) {
            Ok(idx) => array
                .patch_values()
                .execute_scalar(start + idx, ctx)?
                .cast(array.dtype()),
            Err(_) => array.inner().execute_scalar(index, ctx),
        }
    }
}

/// The patch ordinals owned by `chunk`, read in place when the offsets are canonical.
fn chunk_bounds(
    array: ArrayView<'_, Patched>,
    chunk: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(usize, usize)> {
    if let Some(offsets) = array.chunk_offsets().as_opt::<Primitive>() {
        let offsets = offsets.as_slice::<u32>();
        return Ok((offsets[chunk] as usize, offsets[chunk + 1] as usize));
    }
    let bound = |idx: usize, ctx: &mut ExecutionCtx| -> VortexResult<usize> {
        array
            .chunk_offsets()
            .execute_scalar(idx, ctx)?
            .as_primitive()
            .as_::<usize>()
            .ok_or_else(|| vortex_err!("chunk offset does not fit in usize"))
    };
    Ok((bound(chunk, ctx)?, bound(chunk + 1, ctx)?))
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Patched;
    use crate::dtype::Nullability;
    use crate::optimizer::ArrayOptimizer;
    use crate::patches::Patches;
    use crate::scalar::Scalar;

    #[test]
    fn test_simple() -> VortexResult<()> {
        let values = buffer![0u16; 1024].into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
        )?;

        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)?.into_array();

        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(
            array.execute_scalar(0, &mut ctx)?,
            Scalar::primitive(0u16, Nullability::NonNullable)
        );
        for index in 1..=3 {
            assert_eq!(
                array.execute_scalar(index, &mut ctx)?,
                Scalar::primitive(1u16, Nullability::NonNullable)
            );
        }
        Ok(())
    }

    #[test]
    fn test_multi_chunk() -> VortexResult<()> {
        let values = buffer![0u16; 4096].into_array();
        let patches = Patches::new(
            4096,
            0,
            buffer![1u32, 2, 3, 1500, 4095].into_array(),
            buffer![1u16; 5].into_array(),
            None,
        )?;

        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)?.into_array();

        let mut ctx = array_session().create_execution_ctx();
        for index in 0..array.len() {
            let value = array.execute_scalar(index, &mut ctx)?;
            if [1, 2, 3, 1500, 4095].contains(&index) {
                assert_eq!(value, 1u16.into());
            } else {
                assert_eq!(value, 0u16.into());
            }
        }
        Ok(())
    }

    #[test]
    fn test_multi_chunk_sliced() -> VortexResult<()> {
        let values = buffer![0u16; 4096].into_array();
        let patches = Patches::new(
            4096,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
        )?;

        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)?
            .into_array()
            .slice(3..4096)?
            .optimize()?;

        assert!(array.is::<Patched>());

        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(array.execute_scalar(0, &mut ctx)?, 1u16.into());
        for index in 1..array.len() {
            assert_eq!(array.execute_scalar(index, &mut ctx)?, 0u16.into());
        }
        Ok(())
    }
}

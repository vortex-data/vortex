// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use crate::PATCH_SLOTS;
use crate::Sparse;
use crate::SparseExt as _;

impl OperationsVTable<Sparse> for Sparse {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Sparse>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(state.array().dtype().clone()));
        }
        patched_or_fill(state, index, ctx)
    }

    fn scalar_at(
        array: ArrayView<'_, Sparse>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        patched_or_fill(&mut ProbeState::once(array), index, ctx)
    }
}

/// The patch at `index` if there is one, otherwise the fill value.
fn patched_or_fill(
    state: &mut ProbeState<'_, Sparse>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let array = state.array();
    Ok(array
        .patches()
        .probe_patched(state, PATCH_SLOTS, index, ctx)?
        .unwrap_or_else(|| array.fill_scalar().clone()))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Sparse;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn slice_partially_invalid() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = buffer![0u64].into_array();
        let indices = buffer![0u8].into_array();

        let sparse = Sparse::try_new(indices, values, 1000, 999u64.into()).unwrap();
        let sliced = sparse.slice(0..1000).unwrap();
        let mut expected = vec![999u64; 1000];
        expected[0] = 0;

        let values = sliced.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert_arrays_eq!(values, PrimitiveArray::from_iter(expected), &mut ctx);
    }

    /// Indices that are not a canonical primitive array are searched through the repeated
    /// probe's indices slot, which every read must find the same patches through.
    #[test]
    fn repeated_probe_with_encoded_indices() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let indices = ChunkedArray::try_new(
            vec![buffer![1u32, 4].into_array(), buffer![7u32, 9].into_array()],
            DType::Primitive(PType::U32, Nullability::NonNullable),
        )?
        .into_array();
        let values = buffer![10i32, 40, 70, 90].into_array();
        let sparse = Sparse::try_new(indices, values, 12, Scalar::from(-1i32))?.into_array();

        let mut probe = sparse.repeated_probe();
        let read = (0..sparse.len())
            .map(|idx| {
                Ok(probe
                    .execute_scalar(idx, &mut ctx)?
                    .as_primitive()
                    .as_::<i32>())
            })
            .collect::<VortexResult<Vec<_>>>()?;
        assert_eq!(
            read,
            [-1, 10, -1, -1, 40, -1, -1, 70, -1, 90, -1, -1].map(Some)
        );
        Ok(())
    }
}

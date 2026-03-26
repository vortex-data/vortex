// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::DynArray;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArray;
use crate::arrays::patched::patch_lanes;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<Patched> for Patched {
    fn scalar_at(
        array: &PatchedArray,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // First check the patches
        let chunk = index / 1024;
        #[allow(clippy::cast_possible_truncation)]
        let chunk_index = (index % 1024) as u16;

        let values_ptype = PType::try_from(array.dtype())?;

        let lane = match_each_native_ptype!(values_ptype, |V| { index % patch_lanes::<V>() });

        let range = array.seek_to_lane(chunk, lane);

        // Get the range of indices corresponding to the lane, potentially decoding them to avoid
        // the overhead of repeated scalar_at calls.
        let patch_indices = array
            .indices
            .slice(range.clone())?
            .optimize()?
            .execute::<PrimitiveArray>(ctx)?;

        // NOTE: we do linear scan as lane has <= 32 patches, binary search would likely
        //  be slower.
        for (&patch_index, index) in std::iter::zip(patch_indices.as_slice::<u16>(), range) {
            if patch_index == chunk_index {
                return array.values.scalar_at(index)?.cast(array.dtype());
            }
        }

        // Otherwise, access the underlying value.
        array.inner.scalar_at(index)
    }
}

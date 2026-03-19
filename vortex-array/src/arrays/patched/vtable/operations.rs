// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::DynArray;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArray;
use crate::arrays::patched::patch_lanes;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<Patched> for Patched {
    fn scalar_at(array: &PatchedArray, index: usize) -> VortexResult<Scalar> {
        // First check the patches
        let chunk = index / 1024;
        #[allow(clippy::cast_possible_truncation)]
        let chunk_index = (index % 1024) as u16;

        let values_ptype = PType::try_from(array.dtype())?;

        let lane = match_each_native_ptype!(values_ptype, |V| { index % patch_lanes::<V>() });
        let accessor = array.accessor();

        // NOTE: we do linear scan as lane has <= 32 patches, binary search would likely
        //  be slower.
        for (index, patch_index) in accessor.offsets_iter(chunk, lane) {
            if patch_index == chunk_index {
                return array.values.scalar_at(index);
            }
        }

        // Otherwise, access the underlying value.
        array.inner.scalar_at(index)
    }
}

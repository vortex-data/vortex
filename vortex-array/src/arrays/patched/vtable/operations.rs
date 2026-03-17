// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::DynArray;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArray;
use crate::arrays::patched::patch_lanes;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<Patched> for Patched {
    fn scalar_at(array: &PatchedArray, index: usize) -> VortexResult<Scalar> {
        // First check the patches
        let chunk = index / 1024;
        #[allow(clippy::cast_possible_truncation)]
        let chunk_index = (index % 1024) as u16;
        match_each_native_ptype!(array.values_ptype, |V| {
            let lane = index % patch_lanes::<V>();
            let accessor = array.accessor::<V>();
            let patches = accessor.access(chunk, lane);
            // NOTE: we do linear scan as lane has <= 32 patches, binary search would likely
            //  be slower.
            for (patch_index, patch_value) in patches.iter() {
                if patch_index == chunk_index {
                    return Ok(Scalar::primitive(
                        patch_value,
                        array.inner.dtype().nullability(),
                    ));
                }
            }
        });

        // Otherwise, access the underlying value.
        array.inner.scalar_at(index)
    }
}

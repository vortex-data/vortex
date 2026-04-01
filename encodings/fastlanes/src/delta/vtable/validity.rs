// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexResult;

use crate::Delta;
use crate::DeltaArray;
use crate::bit_transpose::untranspose_validity;

impl ValidityVTable<Delta> for Delta {
    fn validity(array: &DeltaArray) -> VortexResult<Validity> {
        let start = array.offset();
        let end = start + array.len();

        let validity = untranspose_validity(
            &array.deltas().validity()?,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        validity.slice(start..end)
    }
}

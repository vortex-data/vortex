// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::VortexSessionExecute;
use vortex_array::legacy_session;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexResult;

use crate::Delta;
use crate::bit_transpose::untranspose_validity;
use crate::delta::array::DeltaArrayExt;

impl ValidityVTable<Delta> for Delta {
    #[allow(clippy::disallowed_methods)]
    fn validity(array: ArrayView<'_, Delta>) -> VortexResult<Validity> {
        let start = array.offset();
        let end = start + array.len();

        let validity = untranspose_validity(
            &array.deltas().validity()?,
            &mut legacy_session().create_execution_ctx(),
        )?;
        validity.slice(start..end)
    }
}

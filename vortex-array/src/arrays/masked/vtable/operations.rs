// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::DynArray;
use crate::arrays::MaskedVTable;
use crate::arrays::masked::MaskedArray;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<MaskedVTable> for MaskedVTable {
    fn scalar_at(array: &MaskedArray, index: usize) -> VortexResult<Scalar> {
        // Invalid indices are handled by the entrypoint function.
        Ok(array.child.scalar_at(index)?.into_nullable())
    }
}

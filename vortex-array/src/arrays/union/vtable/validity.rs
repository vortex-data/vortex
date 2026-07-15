// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::Union;
use crate::validity::Validity;

impl ValidityVTable<Union> for Union {
    fn validity(_array: ArrayView<'_, Union>) -> VortexResult<Validity> {
        // TODO(connor)[Union]: define how union and child nullability interact.
        Ok(Validity::NonNullable)
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexResult;

use super::EliasFano;

impl ValidityVTable<EliasFano> for EliasFano {
    fn validity(_array: ArrayView<'_, EliasFano>) -> VortexResult<Validity> {
        // No validity yet, for simplicity. See `EliasFano` for how to add it backwards compatibly.
        Ok(Validity::NonNullable)
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexResult;

use super::Unary;

impl ValidityVTable<Unary> for Unary {
    fn validity(_array: ArrayView<'_, Unary>) -> VortexResult<Validity> {
        // No validity yet, for simplicity. See `Unary` for how to add it backwards compatibly.
        Ok(Validity::NonNullable)
    }
}

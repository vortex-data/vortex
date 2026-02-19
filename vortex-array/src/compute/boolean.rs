// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;

/// Point-wise Kleene logical _and_ between two Boolean arrays.
#[deprecated(note = "use expr::and_kleene instead")]
pub fn and_kleene(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    crate::expr::and_kleene(lhs, rhs)
}

/// Point-wise Kleene logical _or_ between two Boolean arrays.
#[deprecated(note = "use expr::or_kleene instead")]
pub fn or_kleene(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    crate::expr::or_kleene(lhs, rhs)
}

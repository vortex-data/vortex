// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::builtins::ArrayBuiltins;

/// Logically invert a boolean array, preserving its validity.
#[deprecated(note = "use array.not() via ArrayBuiltins instead")]
pub fn invert(array: &ArrayRef) -> VortexResult<ArrayRef> {
    array.to_array().not()
}

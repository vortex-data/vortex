// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::builtins::ArrayBuiltins;

/// Compute a `Bool`-typed array the same length as `array` where elements is `true` if the list
/// item contains the `value`, `false` otherwise.
///
/// **Deprecated**: Use `array.list_contains(value)` via [`crate::builtins::ArrayBuiltins`] instead.
#[deprecated(note = "Use `array.list_contains(value)` via `ArrayBuiltins` instead")]
pub fn list_contains(array: &ArrayRef, value: &ArrayRef) -> VortexResult<ArrayRef> {
    array.to_array().list_contains(value.to_array())
}

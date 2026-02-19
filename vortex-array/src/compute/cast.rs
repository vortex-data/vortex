// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::dtype::DType;

/// Cast an array to the given data type.
///
/// Use [`crate::builtins::ArrayBuiltins::cast`] instead.
#[deprecated(note = "Use `array.cast(dtype)` via `ArrayBuiltins` trait instead")]
pub fn cast(array: &dyn super::Array, dtype: &DType) -> VortexResult<ArrayRef> {
    use crate::builtins::ArrayBuiltins as _;
    array.to_array().cast(dtype.clone())
}

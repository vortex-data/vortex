// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::builtins::ArrayBuiltins;

/// Performs element-wise conditional selection between two arrays based on a mask.
///
/// Returns a new array where `result[i] = if_true[i]` when `mask[i]` is true,
/// otherwise `result[i] = if_false[i]`.
///
/// Null values in the mask are treated as false (selecting `if_false`). This follows
/// SQL semantics (DuckDB, Trino) where a null condition falls through to the ELSE branch,
/// rather than Arrow's `if_else` which propagates null conditions to the output.
#[deprecated(note = "use if_true.zip(if_false, mask) via ArrayBuiltins instead")]
pub fn zip(if_true: &ArrayRef, if_false: &ArrayRef, mask: &Mask) -> VortexResult<ArrayRef> {
    if_true
        .to_array()
        .zip(if_false.to_array(), mask.clone().into_array())
}

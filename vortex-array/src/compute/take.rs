// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;

/// Creates a new array using the elements from the input `array` indexed by `indices`.
///
/// For example, if we have an `array` `[1, 2, 3, 4, 5]` and `indices` `[4, 2]`, the resulting
/// array would be `[5, 3]`.
///
/// The output array will have the same length as the `indices` array.
pub fn take(array: &dyn Array, indices: &dyn Array) -> VortexResult<ArrayRef> {
    // TODO(joe): inline usage and remove to_canonical().
    array
        .take(indices.to_array())?
        .to_canonical()
        .map(|c| c.into_array())
}

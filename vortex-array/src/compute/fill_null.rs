// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::Array;
use crate::ArrayRef;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;

/// Replace nulls in the array with another value.
///
/// # Examples
///
/// ```
/// use vortex_array::IntoArray;
/// use vortex_array::arrays::{PrimitiveArray};
/// use vortex_array::compute::{fill_null};
/// use vortex_array::scalar::Scalar;
///
/// let array =
///     PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)]);
/// let array = fill_null(&array.into_array(), &Scalar::from(42i32)).unwrap();
/// assert_eq!(array.display_values().to_string(), "[0i32, 42i32, 1i32, 42i32, 2i32]");
/// ```
#[deprecated(note = "use array.fill_null(scalar) via ArrayBuiltins instead")]
pub fn fill_null(array: &ArrayRef, fill_value: &Scalar) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        !fill_value.is_null(),
        "fill_null requires a non-null fill value"
    );
    let result = array.to_array().fill_null(fill_value.clone())?;
    debug_assert!(
        fill_value.dtype().is_nullable() || !result.dtype().is_nullable(),
        "fill_null with non-nullable fill value must produce a non-nullable result"
    );
    Ok(result)
}

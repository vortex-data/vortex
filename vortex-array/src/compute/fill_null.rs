// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::builtins::ArrayBuiltins;

/// Replace nulls in the array with another value.
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::{PrimitiveArray};
/// use vortex_array::compute::{fill_null};
/// use vortex_scalar::Scalar;
///
/// let array =
///     PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)]);
/// let array = fill_null(array.as_ref(), &Scalar::from(42i32)).unwrap();
/// assert_eq!(array.display_values().to_string(), "[0i32, 42i32, 1i32, 42i32, 2i32]");
/// ```
#[deprecated(note = "use array.fill_null(scalar) via ArrayBuiltins instead")]
pub fn fill_null(array: &dyn Array, fill_value: &Scalar) -> VortexResult<ArrayRef> {
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

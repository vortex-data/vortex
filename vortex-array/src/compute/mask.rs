// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::expr::EmptyOptions;
use crate::expr::mask::Mask as MaskExpr;
use crate::validity::Validity;

/// Replace values with null where the mask is true.
///
/// The returned array is nullable but otherwise has the same dtype and length as `array`.
///
/// This function returns a lazy `ScalarFnArray` wrapping the [`Mask`](crate::expr::mask::Mask)
/// expression that defers the actual masking operation until execution time. The mask is inverted
/// (true=mask-out becomes true=keep) and passed as a boolean child to the expression.
///
/// # Examples
///
/// ```
/// use vortex_array::IntoArray;
/// use vortex_array::arrays::{BoolArray, PrimitiveArray};
/// use vortex_array::compute::{ mask};
/// use vortex_error::VortexResult;
/// use vortex_mask::Mask;
/// use vortex_scalar::Scalar;
///
/// # fn main() -> VortexResult<()> {
/// let array =
///     PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)]);
/// let mask_array = Mask::from_iter([true, false, false, false, true]);
///
/// let masked = mask(array.as_ref(), &mask_array)?;
/// assert_eq!(masked.len(), 5);
/// assert!(!masked.is_valid(0).unwrap());
/// assert!(!masked.is_valid(1).unwrap());
/// assert_eq!(masked.scalar_at(2)?, Scalar::from(Some(1)));
/// assert!(!masked.is_valid(3).unwrap());
/// assert!(!masked.is_valid(4).unwrap());
/// # Ok(())
/// # }
/// ```
///
pub fn mask(array: &dyn Array, mask: &Mask) -> VortexResult<ArrayRef> {
    let mask_true_count = mask.true_count();

    if mask_true_count == 0 {
        // Fast-path for empty mask: nothing to mask out.
        return array.to_array().cast(array.dtype().as_nullable());
    }

    if mask_true_count == mask.len() {
        // Fast-path for full mask: everything is masked out.
        return Ok(
            ConstantArray::new(Scalar::null(array.dtype().as_nullable()), array.len()).into_array(),
        );
    }

    // Do nothing if the array is already all nulls.
    if array.all_invalid()? {
        return Ok(array.to_array());
    }

    // Lazy wrap: invert the mask (true=mask_out → true=keep) and create a ScalarFnArray
    // wrapping the Mask expression.
    let keep_mask = BoolArray::new(mask.to_bit_buffer().not(), Validity::NonNullable);
    MaskExpr.try_new_array(
        array.len(),
        EmptyOptions,
        [array.to_array(), keep_mask.into_array()],
    )
}

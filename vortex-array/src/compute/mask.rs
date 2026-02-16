// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::builtins::ArrayBuiltins;

/// Replace values with null where the mask is true.
///
/// The returned array is nullable but otherwise has the same dtype and length as `array`.
///
/// This function returns a lazy `ScalarFnArray` wrapping the [`Mask`](crate::expr::mask::Mask)
/// expression that defers the actual masking operation until execution time. The mask is inverted
/// (true=mask-out becomes true=keep) and passed as a boolean child to the expression.
pub fn mask(array: &dyn Array, mask: &Mask) -> VortexResult<ArrayRef> {
    match mask {
        Mask::AllTrue(_) => Ok(array.to_array()),
        Mask::AllFalse(_) => Ok(ConstantArray::new(
            Scalar::null(array.dtype().as_nullable()),
            array.len(),
        )
        .into_array()),
        Mask::Values(val) => array.to_array().mask(val.into_array()),
    }
}

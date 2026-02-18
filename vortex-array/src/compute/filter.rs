// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// TODO(connor): REMOVE THIS FILE!

use arrow_array::BooleanArray;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrow::FromArrowArray;
use crate::arrow::IntoArrowArray;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;

/// Keep only the elements for which the corresponding mask value is true.
///
/// # Examples
///
/// ```
/// use vortex_array::{Array, IntoArray};
/// use vortex_array::arrays::{BoolArray, PrimitiveArray};
/// use vortex_array::compute::{ filter, mask};
/// use vortex_error::VortexResult;
/// use vortex_mask::Mask;
/// use vortex_array::scalar::Scalar;
///
/// # fn main() -> VortexResult<()> {
/// let array =
///     PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)]);
/// let mask = Mask::from_iter([true, false, false, false, true]);
///
/// let filtered = filter(array.as_ref(), &mask)?;
/// assert_eq!(filtered.len(), 2);
/// assert_eq!(filtered.scalar_at(0)?, Scalar::from(Some(0_i32)));
/// assert_eq!(filtered.scalar_at(1)?, Scalar::from(Some(2_i32)));
/// # Ok(())
/// # }
/// ```
///
/// # Panics
///
/// The `predicate` must receive an Array with type non-nullable bool, and will panic if this is
/// not the case.
pub fn filter(array: &dyn Array, mask: &Mask) -> VortexResult<ArrayRef> {
    // TODO(connor): Remove this function completely!!!
    Ok(array.filter(mask.clone())?.to_canonical()?.into_array())
}

impl dyn Array + '_ {
    /// Converts from a possible nullable boolean array. Null values are treated as false.
    pub fn try_to_mask_fill_null_false(&self) -> VortexResult<Mask> {
        if !matches!(self.dtype(), DType::Bool(_)) {
            vortex_bail!("mask must be bool array, has dtype {}", self.dtype());
        }

        // Convert nulls to false first in case this can be done cheaply by the encoding.
        let array = self
            .to_array()
            .fill_null(Scalar::bool(false, self.dtype().nullability()))?;

        Ok(array.to_bool().to_mask_fill_null_false())
    }
}

pub fn arrow_filter_fn(array: &dyn Array, mask: &Mask) -> VortexResult<ArrayRef> {
    let values = match &mask {
        Mask::Values(values) => values,
        Mask::AllTrue(_) | Mask::AllFalse(_) => unreachable!("check in filter invoke"),
    };

    let array_ref = array.to_array().into_arrow_preferred()?;
    let mask_array = BooleanArray::new(values.bit_buffer().clone().into(), None);
    let filtered = arrow_select::filter::filter(array_ref.as_ref(), &mask_array)?;

    ArrayRef::from_arrow(filtered.as_ref(), array.dtype().is_nullable())
}

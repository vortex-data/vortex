use std::ops::BitAnd;

use arrow_array::BooleanArray;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexError, VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::array::ConstantArray;
use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::compute::scalar_at;
use crate::encoding::Encoding;
use crate::{Array, Canonical, IntoArray, IntoArrayVariant};

pub trait FilterFn<A> {
    /// Filter an array by the provided predicate.
    ///
    /// Note that the entry-point filter functions handles `Mask::AllTrue` and `Mask::AllFalse`,
    /// leaving only `Mask::Values` to be handled by this function.
    fn filter(&self, array: &A, mask: &Mask) -> VortexResult<Array>;
}

impl<E: Encoding> FilterFn<Array> for E
where
    E: FilterFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn filter(&self, array: &Array, mask: &Mask) -> VortexResult<Array> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        FilterFn::filter(encoding, array_ref, mask)
    }
}

/// Return a new array by applying a boolean predicate to select items from a base Array.
///
/// # Performance
///
/// This function attempts to amortize the cost of copying
///
/// # Panics
///
/// The `predicate` must receive an Array with type non-nullable bool, and will panic if this is
/// not the case.
pub fn filter(array: &Array, mask: &Mask) -> VortexResult<Array> {
    if mask.len() != array.len() {
        vortex_bail!(
            "mask.len() is {}, does not equal array.len() of {}",
            mask.len(),
            array.len()
        );
    }

    let true_count = mask.true_count();

    // Fast-path for empty mask.
    if true_count == 0 {
        return Ok(Canonical::empty(array.dtype()).into());
    }

    // Fast-path for full mask
    if true_count == mask.len() {
        return Ok(array.clone());
    }

    let filtered = filter_impl(array, mask)?;

    debug_assert_eq!(
        filtered.len(),
        true_count,
        "Filter length mismatch {}",
        array.encoding()
    );
    debug_assert_eq!(
        filtered.dtype(),
        array.dtype(),
        "Filter dtype mismatch {}",
        array.encoding()
    );

    Ok(filtered)
}

fn filter_impl(array: &Array, mask: &Mask) -> VortexResult<Array> {
    // Since we handle the AllTrue and AllFalse cases in the entry-point filter function,
    // implementations can use `AllOr::expect_some` to unwrap the mixed values variant.
    let values = match &mask {
        Mask::AllTrue(_) => return Ok(array.clone()),
        Mask::AllFalse(_) => return Ok(Canonical::empty(array.dtype()).into_array()),
        Mask::Values(values) => values,
    };

    if let Some(filter_fn) = array.vtable().filter_fn() {
        let result = filter_fn.filter(array, mask)?;
        debug_assert_eq!(result.len(), mask.true_count());
        return Ok(result);
    }

    // We can use scalar_at if the mask has length 1.
    if mask.true_count() == 1 && array.vtable().scalar_at_fn().is_some() {
        let idx = mask.first().vortex_expect("true_count == 1");
        return Ok(ConstantArray::new(scalar_at(array, idx)?, 1).into_array());
    }

    // Fallback: implement using Arrow kernels.
    log::debug!("No filter implementation found for {}", array.encoding(),);

    let array_ref = array.clone().into_arrow_preferred()?;
    let mask_array = BooleanArray::new(values.boolean_buffer().clone(), None);
    let filtered = arrow_select::filter::filter(array_ref.as_ref(), &mask_array)?;

    Ok(Array::from_arrow(filtered, array.dtype().is_nullable()))
}

impl TryFrom<Array> for Mask {
    type Error = VortexError;

    /// Converts from a possible nullable boolean array. Null values are treated as false.
    fn try_from(array: Array) -> Result<Self, Self::Error> {
        if !matches!(array.dtype(), DType::Bool(_)) {
            vortex_bail!("mask must be bool array, has dtype {}", array.dtype());
        }

        if let Some(constant) = array.as_constant() {
            let bool_constant = constant.as_bool();
            if bool_constant.value().unwrap_or(false) {
                return Ok(Self::new_true(array.len()));
            } else {
                return Ok(Self::new_false(array.len()));
            }
        }

        // Otherwise, we canonicalize
        let array = array.into_bool()?;

        // Extract a boolean buffer, treating null values to false
        let buffer = match array.validity_mask()? {
            Mask::AllTrue(_) => array.boolean_buffer(),
            Mask::AllFalse(_) => return Ok(Self::new_false(array.len())),
            Mask::Values(validity) => validity.boolean_buffer().bitand(&array.boolean_buffer()),
        };

        Ok(Self::from_buffer(buffer))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::array::{BoolArray, PrimitiveArray};
    use crate::compute::filter::filter;
    use crate::IntoArray;

    #[test]
    fn test_filter() {
        let items =
            PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)])
                .into_array();
        let mask =
            Mask::try_from(BoolArray::from_iter([true, false, true, false, true]).into_array())
                .unwrap();

        let filtered = filter(&items, &mask).unwrap();
        assert_eq!(
            filtered.into_primitive().unwrap().as_slice::<i32>(),
            &[0i32, 1i32, 2i32]
        );
    }
}

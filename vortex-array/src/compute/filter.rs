use std::cmp::Ordering;
use std::ops::BitAnd;
use std::sync::{Arc, OnceLock};

use arrow_array::BooleanArray;
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_panic, VortexError, VortexExpect, VortexResult};
use vortex_mask::{Mask, MaskValues};

use crate::array::{BoolArray, ConstantArray};
use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::compute::scalar_at;
use crate::encoding::Encoding;
use crate::stats::Stat;
use crate::{ArrayData, Canonical, IntoArrayData, IntoArrayVariant};

pub trait FilterFn<Array> {
    /// Filter an array by the provided predicate.
    ///
    /// Note that the entry-point filter functions handles `Mask::AllTrue` and `Mask::AllFalse`,
    /// leaving only `Mask::Values` to be handled by this function.
    fn filter(&self, array: &Array, mask: &Mask) -> VortexResult<ArrayData>;
}

impl<E: Encoding> FilterFn<ArrayData> for E
where
    E: FilterFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn filter(&self, array: &ArrayData, mask: &Mask) -> VortexResult<ArrayData> {
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
pub fn filter(array: &ArrayData, mask: &Mask) -> VortexResult<ArrayData> {
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

fn filter_impl(array: &ArrayData, mask: &Mask) -> VortexResult<ArrayData> {
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

    Ok(ArrayData::from_arrow(filtered, array.dtype().is_nullable()))
}

impl TryFrom<ArrayData> for Mask {
    type Error = VortexError;

    fn try_from(array: ArrayData) -> Result<Self, Self::Error> {
        if array.dtype() != &DType::Bool(Nullability::NonNullable) {
            vortex_bail!(
                "mask must be non-nullable bool, has dtype {}",
                array.dtype(),
            );
        }

        if let Some(true_count) = array.statistics().get_as::<u64>(Stat::TrueCount) {
            let len = array.len();
            if true_count == 0 {
                return Ok(Self::new_false(len));
            }
            if true_count == len as u64 {
                return Ok(Self::new_true(len));
            }
        }

        // TODO(ngates): should we have a `to_filter_mask` compute function where encodings
        //  pick the best possible conversion? E.g. SparseArray may want from_indices.
        Ok(Self::from_buffer(array.into_bool()?.boolean_buffer()))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::array::{BoolArray, PrimitiveArray};
    use crate::compute::filter::filter;
    use crate::IntoArrayData;

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

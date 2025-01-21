use arrow_array::BooleanArray;
use vortex_error::{vortex_bail, VortexError, VortexResult};
use vortex_scalar::Scalar;

use super::FilterMask;
use crate::array::ConstantArray;
use crate::arrow::FromArrowArray;
use crate::compute::try_cast;
use crate::encoding::Encoding;
use crate::{ArrayDType, ArrayData, IntoArrayData, IntoCanonical};

pub trait MaskFn<Array> {
    /// Replace masked values with null in array.
    fn mask(&self, array: &Array, mask: FilterMask) -> VortexResult<ArrayData>;
}

impl<E: Encoding> MaskFn<ArrayData> for E
where
    E: MaskFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn mask(&self, array: &ArrayData, mask: FilterMask) -> VortexResult<ArrayData> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        MaskFn::mask(encoding, array_ref, mask)
    }
}

/// Replace values with null where the mask is true.
///
/// The returned array is nullable but otherwise has the same dtype and length as `array`.
///
/// # Examples
///
/// ```
/// use vortex_array::IntoArrayData;
/// use vortex_array::array::{BoolArray, PrimitiveArray};
/// use vortex_array::compute::{FilterMask, scalar_at};
/// use vortex_array::compute::mask;
/// use vortex_array::validity::ArrayValidity;
/// use vortex_scalar::Scalar;
///
/// let array =
///     PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)])
///         .into_array();
/// let mask_array = FilterMask::try_from(
///     BoolArray::from_iter([true, false, false, false, true]).into_array(),
/// )
/// .unwrap();
///
/// let masked = mask(&array, mask_array).unwrap();
/// assert_eq!(masked.len(), 5);
/// assert!(!masked.is_valid(0));
/// assert!(!masked.is_valid(1));
/// assert_eq!(scalar_at(&masked, 2).unwrap(), Scalar::from(Some(1)));
/// assert!(!masked.is_valid(3));
/// assert!(!masked.is_valid(4));
/// ```
///
pub fn mask(array: &ArrayData, mask: FilterMask) -> VortexResult<ArrayData> {
    if mask.len() != array.len() {
        vortex_bail!(
            "mask.len() is {}, does not equal array.len() of {}",
            mask.len(),
            array.len()
        );
    }

    let true_count = mask.true_count();

    let masked = if true_count == 0 {
        // Fast-path for empty mask
        try_cast(array, &array.dtype().as_nullable())?
    } else if true_count == mask.len() {
        // Fast-path for full mask.
        ConstantArray::new(
            Scalar::null(array.dtype().clone().as_nullable()),
            array.len(),
        )
        .into_array()
    } else {
        mask_impl(array, mask)?
    };

    debug_assert_eq!(
        masked.len(),
        array.len(),
        "Mask should not change length {}\n\n{:?}\n\n{:?}",
        array.encoding().id(),
        array,
        masked
    );
    debug_assert_eq!(
        masked.dtype(),
        &array.dtype().as_nullable(),
        "Mask dtype mismatch {} {} {} {}",
        array.encoding().id(),
        masked.dtype(),
        array.dtype(),
        array.dtype().as_nullable(),
    );

    Ok(masked)
}

fn mask_impl(array: &ArrayData, mask: FilterMask) -> VortexResult<ArrayData> {
    if let Some(mask_fn) = array.encoding().mask_fn() {
        return mask_fn.mask(array, mask);
    }

    // Fallback: implement using Arrow kernels.
    log::debug!("No mask implementation found for {}", array.encoding().id(),);

    let array_ref = array.clone().into_arrow()?;
    let mask = BooleanArray::new(mask.boolean_buffer().clone(), None);

    let masked = arrow_select::nullif::nullif(array_ref.as_ref(), &mask)?;

    Ok(ArrayData::from_arrow(masked, true))
}

#[cfg(feature = "test-harness")]
pub mod test_harness {
    use crate::array::BoolArray;
    use crate::compute::{mask, scalar_at, FilterMask};
    use crate::validity::ArrayValidity as _;
    use crate::{ArrayData, IntoArrayData};

    pub fn test_mask(array: ArrayData) {
        assert_eq!(array.len(), 5);
        test_heterogenous_mask(&array);
        test_empty_mask(&array);
        test_full_mask(&array);
    }

    #[allow(clippy::unwrap_used)]
    fn test_heterogenous_mask(array: &ArrayData) {
        let mask_array = FilterMask::try_from(
            BoolArray::from_iter([true, false, false, true, true]).into_array(),
        )
        .unwrap();
        let masked = mask(array, mask_array).unwrap();
        assert_eq!(masked.len(), array.len());
        assert!(!masked.is_valid(0));
        assert_eq!(
            scalar_at(&masked, 1).unwrap(),
            scalar_at(array, 1).unwrap().into_nullable()
        );
        assert_eq!(
            scalar_at(&masked, 2).unwrap(),
            scalar_at(array, 2).unwrap().into_nullable()
        );
        assert!(!masked.is_valid(3));
        assert!(!masked.is_valid(4));
    }

    #[allow(clippy::unwrap_used)]
    fn test_empty_mask(array: &ArrayData) {
        let all_unmasked = FilterMask::try_from(
            BoolArray::from_iter([false, false, false, false, false]).into_array(),
        )
        .unwrap();
        let masked = mask(array, all_unmasked).unwrap();
        assert_eq!(masked.len(), array.len());
        assert_eq!(
            scalar_at(&masked, 0).unwrap(),
            scalar_at(array, 0).unwrap().into_nullable()
        );
        assert_eq!(
            scalar_at(&masked, 1).unwrap(),
            scalar_at(array, 1).unwrap().into_nullable()
        );
        assert_eq!(
            scalar_at(&masked, 2).unwrap(),
            scalar_at(array, 2).unwrap().into_nullable()
        );
        assert_eq!(
            scalar_at(&masked, 3).unwrap(),
            scalar_at(array, 3).unwrap().into_nullable()
        );
        assert_eq!(
            scalar_at(&masked, 4).unwrap(),
            scalar_at(array, 4).unwrap().into_nullable()
        );
    }

    #[allow(clippy::unwrap_used)]
    fn test_full_mask(array: &ArrayData) {
        let all_masked =
            FilterMask::try_from(BoolArray::from_iter([true, true, true, true, true]).into_array())
                .unwrap();
        let masked = mask(array, all_masked).unwrap();
        assert_eq!(masked.len(), array.len());
        assert!(!masked.is_valid(0));
        assert!(!masked.is_valid(1));
        assert!(!masked.is_valid(2));
        assert!(!masked.is_valid(3));
        assert!(!masked.is_valid(4));

        let mask1 = FilterMask::try_from(
            BoolArray::from_iter([true, false, false, true, true]).into_array(),
        )
        .unwrap();
        let mask2 = FilterMask::try_from(
            BoolArray::from_iter([false, true, false, false, true]).into_array(),
        )
        .unwrap();
        let first = mask(array, mask1).unwrap();
        let double_masked = mask(&first, mask2).unwrap();
        assert_eq!(double_masked.len(), array.len());
        assert!(!double_masked.is_valid(0));
        assert!(!double_masked.is_valid(1));
        assert_eq!(
            scalar_at(&double_masked, 2).unwrap(),
            scalar_at(array, 2).unwrap().into_nullable()
        );
        assert!(!double_masked.is_valid(3));
        assert!(!double_masked.is_valid(4));
    }
}

#[cfg(test)]
mod test {
    use super::test_harness::test_mask;
    use crate::array::PrimitiveArray;
    use crate::IntoArrayData as _;

    #[test]
    fn test_mask_non_nullable_array() {
        let non_nullable_array = PrimitiveArray::from_iter([1, 2, 3, 4, 5]).into_array();
        test_mask(non_nullable_array);
    }
}

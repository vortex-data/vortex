use arrow_array::BooleanArray;
use vortex_error::{vortex_bail, VortexError, VortexExpect, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::ConstantArray;
use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::compute::try_cast;
use crate::encoding::Encoding;
use crate::{Array, ArrayRef, IntoArray};

pub trait MaskFn<A> {
    /// Replace masked values with null in array.
    fn mask(&self, array: A, mask: Mask) -> VortexResult<ArrayRef>;
}

impl<E: Encoding> MaskFn<&dyn Array> for E
where
    E: for<'a> MaskFn<&'a E::Array>,
{
    fn mask(&self, array: &dyn Array, mask: Mask) -> VortexResult<ArrayRef> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        let vtable = array.vtable();

        MaskFn::mask(self, array_ref, mask)
    }
}

/// Replace values with null where the mask is true.
///
/// The returned array is nullable but otherwise has the same dtype and length as `array`.
///
/// # Examples
///
/// ```
/// use vortex_array::IntoArray;
/// use vortex_array::arrays::{BoolArray, PrimitiveArray};
/// use vortex_array::compute::{scalar_at, mask};
/// use vortex_mask::Mask;
/// use vortex_scalar::Scalar;
///
/// let array =
///     PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)]);
/// let mask_array = Mask::try_from(
///     &BoolArray::from_iter([true, false, false, false, true]),
/// )
/// .unwrap();
///
/// let masked = mask(&array, mask_array).unwrap();
/// assert_eq!(masked.len(), 5);
/// assert!(!masked.is_valid(0).unwrap());
/// assert!(!masked.is_valid(1).unwrap());
/// assert_eq!(scalar_at(&masked, 2).unwrap(), Scalar::from(Some(1)));
/// assert!(!masked.is_valid(3).unwrap());
/// assert!(!masked.is_valid(4).unwrap());
/// ```
///
pub fn mask(array: &dyn Array, mask: Mask) -> VortexResult<ArrayRef> {
    if mask.len() != array.len() {
        vortex_bail!(
            "mask.len() is {}, does not equal array.len() of {}",
            mask.len(),
            array.len()
        );
    }

    let masked = if matches!(mask, Mask::AllFalse(_)) {
        // Fast-path for empty mask
        try_cast(array, &array.dtype().as_nullable())?
    } else if matches!(mask, Mask::AllTrue(_)) {
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
        array.encoding(),
        array,
        masked
    );
    debug_assert_eq!(
        masked.dtype(),
        &array.dtype().as_nullable(),
        "Mask dtype mismatch {} {} {} {}",
        array.encoding(),
        masked.dtype(),
        array.dtype(),
        array.dtype().as_nullable(),
    );

    Ok(masked)
}

fn mask_impl(array: &dyn Array, mask: Mask) -> VortexResult<ArrayRef> {
    if let Some(mask_fn) = array.vtable().mask_fn() {
        return mask_fn.mask(array, mask);
    }

    // Fallback: implement using Arrow kernels.
    log::debug!("No mask implementation found for {}", array.encoding());

    let array_ref = array.to_array().into_arrow_preferred()?;
    let mask = BooleanArray::new(mask.to_boolean_buffer(), None);

    let masked = arrow_select::nullif::nullif(array_ref.as_ref(), &mask)?;

    Ok(ArrayRef::from_arrow(masked, true))
}

#[cfg(feature = "test-harness")]
pub mod test_harness {
    use vortex_mask::Mask;

    use crate::arrays::BoolArray;
    use crate::compute::{mask, scalar_at};
    use crate::{Array, ArrayRef, IntoArray};

    pub fn test_mask(array: &dyn Array) {
        assert_eq!(array.len(), 5);
        test_heterogenous_mask(array);
        test_empty_mask(array);
        test_full_mask(array);
    }

    #[allow(clippy::unwrap_used)]
    fn test_heterogenous_mask(array: &dyn Array) {
        let mask_array =
            Mask::try_from(&BoolArray::from_iter([true, false, false, true, true])).unwrap();
        let masked = mask(array, mask_array).unwrap();
        assert_eq!(masked.len(), array.len());
        assert!(!masked.is_valid(0).unwrap());
        assert_eq!(
            scalar_at(&masked, 1).unwrap(),
            scalar_at(array, 1).unwrap().into_nullable()
        );
        assert_eq!(
            scalar_at(&masked, 2).unwrap(),
            scalar_at(array, 2).unwrap().into_nullable()
        );
        assert!(!masked.is_valid(3).unwrap());
        assert!(!masked.is_valid(4).unwrap());
    }

    #[allow(clippy::unwrap_used)]
    fn test_empty_mask(array: &dyn Array) {
        let all_unmasked =
            Mask::try_from(&BoolArray::from_iter([false, false, false, false, false])).unwrap();
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
    fn test_full_mask(array: &dyn Array) {
        let all_masked =
            Mask::try_from(&BoolArray::from_iter([true, true, true, true, true])).unwrap();
        let masked = mask(array, all_masked).unwrap();
        assert_eq!(masked.len(), array.len());
        assert!(!masked.is_valid(0).unwrap());
        assert!(!masked.is_valid(1).unwrap());
        assert!(!masked.is_valid(2).unwrap());
        assert!(!masked.is_valid(3).unwrap());
        assert!(!masked.is_valid(4).unwrap());

        let mask1 =
            Mask::try_from(&BoolArray::from_iter([true, false, false, true, true])).unwrap();
        let mask2 =
            Mask::try_from(&BoolArray::from_iter([false, true, false, false, true])).unwrap();
        let first = mask(array, mask1).unwrap();
        let double_masked = mask(&first, mask2).unwrap();
        assert_eq!(double_masked.len(), array.len());
        assert!(!double_masked.is_valid(0).unwrap());
        assert!(!double_masked.is_valid(1).unwrap());
        assert_eq!(
            scalar_at(&double_masked, 2).unwrap(),
            scalar_at(array, 2).unwrap().into_nullable()
        );
        assert!(!double_masked.is_valid(3).unwrap());
        assert!(!double_masked.is_valid(4).unwrap());
    }
}

#[cfg(test)]
mod test {
    use super::test_harness::test_mask;
    use crate::arrays::PrimitiveArray;
    use crate::IntoArray as _;

    #[test]
    fn test_mask_non_nullable_array() {
        let non_nullable_array = PrimitiveArray::from_iter([1, 2, 3, 4, 5]);
        test_mask(&non_nullable_array);
    }
}

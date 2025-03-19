use arrow_array::BooleanArray;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::ConstantArray;
use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::compute::try_cast;
use crate::encoding::Encoding;
use crate::{Array, ArrayRef};

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

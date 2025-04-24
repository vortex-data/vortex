use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::i256;

use crate::Array;
use crate::arrays::decimal::serde::DecimalValueType;
use crate::arrays::{DecimalArray, DecimalEncoding, NativeDecimalType};
use crate::compute::{IsSortedFn, IsSortedIteratorExt};

impl IsSortedFn<&DecimalArray> for DecimalEncoding {
    fn is_sorted(&self, array: &DecimalArray) -> VortexResult<bool> {
        match array.values_type {
            DecimalValueType::I128 => compute_is_sorted::<i128>(array, false),
            DecimalValueType::I256 => compute_is_sorted::<i256>(array, false),
        }
    }

    fn is_strict_sorted(&self, array: &DecimalArray) -> VortexResult<bool> {
        match array.values_type {
            DecimalValueType::I128 => compute_is_sorted::<i128>(array, true),
            DecimalValueType::I256 => compute_is_sorted::<i256>(array, true),
        }
    }
}

fn compute_is_sorted<T: NativeDecimalType>(array: &DecimalArray, strict: bool) -> VortexResult<bool>
where
    dyn Iterator<Item = T>: IsSortedIteratorExt,
{
    match array.validity_mask()? {
        Mask::AllFalse(_) => Ok(!strict),
        Mask::AllTrue(_) => {
            let buf = array.buffer::<T>();
            let iter = buf.iter().copied();

            Ok(if strict {
                IsSortedIteratorExt::is_strict_sorted(iter)
            } else {
                iter.is_sorted()
            })
        }
        Mask::Values(mask_values) => {
            let buf = array.buffer::<T>();

            let iter = mask_values
                .boolean_buffer()
                .set_indices()
                .map(|idx| buf[idx]);

            Ok(if strict {
                IsSortedIteratorExt::is_strict_sorted(iter)
            } else {
                iter.is_sorted()
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DecimalDType;

    use crate::arrays::DecimalArray;
    use crate::compute::{is_sorted, is_strict_sorted};
    use crate::validity::Validity;

    #[test]
    fn test_is_sorted() {
        let sorted = buffer![100i128, 200i128, 200i128];
        let unsorted = buffer![200i128, 100i128, 200i128];

        let dtype = DecimalDType::new(19, 2);

        let sorted_array = DecimalArray::new(sorted, dtype, Validity::NonNullable);
        let unsorted_array = DecimalArray::new(unsorted, dtype, Validity::NonNullable);

        assert!(is_sorted(&sorted_array).unwrap());
        assert!(!is_sorted(&unsorted_array).unwrap());
    }

    #[test]
    fn test_is_strict_sorted() {
        let strict_sorted = buffer![100i128, 200i128, 300i128];
        let sorted = buffer![100i128, 200i128, 200i128];

        let dtype = DecimalDType::new(19, 2);

        let strict_sorted_array = DecimalArray::new(strict_sorted, dtype, Validity::NonNullable);
        let sorted_array = DecimalArray::new(sorted, dtype, Validity::NonNullable);

        assert!(is_strict_sorted(&strict_sorted_array).unwrap());
        assert!(!is_strict_sorted(&sorted_array).unwrap());
    }
}

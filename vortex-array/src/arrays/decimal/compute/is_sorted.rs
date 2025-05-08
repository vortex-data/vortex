use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::{NativeDecimalType, match_each_decimal_value_type};

use crate::arrays::{DecimalArray, DecimalEncoding};
use crate::compute::{IsSortedIteratorExt, IsSortedKernel, IsSortedKernelAdapter};
use crate::{Array, register_kernel};

impl IsSortedKernel for DecimalEncoding {
    fn is_sorted(&self, array: &DecimalArray) -> VortexResult<bool> {
        is_decimal_sorted(array, false)
    }

    fn is_strict_sorted(&self, array: &DecimalArray) -> VortexResult<bool> {
        is_decimal_sorted(array, true)
    }
}

register_kernel!(IsSortedKernelAdapter(DecimalEncoding).lift());

fn is_decimal_sorted(array: &DecimalArray, strict: bool) -> VortexResult<bool> {
    match_each_decimal_value_type!(array.values_type, |$S| {
       compute_is_sorted::<$S>(array, strict)
    })
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
            let values = array.buffer::<T>();
            let iter = mask_values
                .boolean_buffer()
                .iter()
                .zip_eq(values)
                .map(|(is_valid, v)| is_valid.then_some(v));

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
    use arrow_array::types::Decimal128Type;
    use arrow_cast::parse::parse_decimal;
    use vortex_buffer::buffer;
    use vortex_dtype::DecimalDType;

    use crate::arrays::DecimalArray;
    use crate::compute::{is_sorted, is_strict_sorted};
    use crate::validity::Validity;

    #[test]
    fn test_is_sorted() {
        let dtype = DecimalDType::new(19, 2);
        let i100 =
            parse_decimal::<Decimal128Type>("100.00", dtype.precision(), dtype.scale()).unwrap();
        let i200 =
            parse_decimal::<Decimal128Type>("200.00", dtype.precision(), dtype.scale()).unwrap();

        let sorted = buffer![i100, i200, i200];
        let unsorted = buffer![i200, i100, i200];

        let sorted_array = DecimalArray::new(sorted, dtype, Validity::NonNullable);
        let unsorted_array = DecimalArray::new(unsorted, dtype, Validity::NonNullable);

        assert!(is_sorted(&sorted_array).unwrap());
        assert!(!is_sorted(&unsorted_array).unwrap());
    }

    #[test]
    fn test_is_strict_sorted() {
        let dtype = DecimalDType::new(19, 2);
        let i100 =
            parse_decimal::<Decimal128Type>("100.00", dtype.precision(), dtype.scale()).unwrap();
        let i200 =
            parse_decimal::<Decimal128Type>("200.00", dtype.precision(), dtype.scale()).unwrap();
        let i300 =
            parse_decimal::<Decimal128Type>("300.00", dtype.precision(), dtype.scale()).unwrap();

        let strict_sorted = buffer![i100, i200, i300];
        let sorted = buffer![i100, i200, i200];

        let dtype = DecimalDType::new(19, 2);

        let strict_sorted_array = DecimalArray::new(strict_sorted, dtype, Validity::NonNullable);
        let sorted_array = DecimalArray::new(sorted, dtype, Validity::NonNullable);

        assert!(is_strict_sorted(&strict_sorted_array).unwrap());
        assert!(!is_strict_sorted(&sorted_array).unwrap());
    }
}

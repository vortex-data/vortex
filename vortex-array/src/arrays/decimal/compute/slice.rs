use vortex_buffer::Buffer;
use vortex_dtype::DecimalDType;
use vortex_error::VortexResult;

use crate::arrays::{DecimalArray, DecimalEncoding, NativeDecimalType};
use crate::compute::SliceFn;
use crate::validity::Validity;
use crate::{Array, ArrayRef, match_each_decimal_value_type};

impl SliceFn<&DecimalArray> for DecimalEncoding {
    fn slice(&self, array: &DecimalArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let sliced = match_each_decimal_value_type!(array.values_type, |$S| {
            slice_typed(
                array.buffer::<$S>(),
                start,
                stop,
                array.decimal_dtype(),
                array.validity.clone(),
            )
        });

        Ok(sliced)
    }
}

fn slice_typed<T: NativeDecimalType>(
    values: Buffer<T>,
    start: usize,
    end: usize,
    decimal_dtype: DecimalDType,
    validity: Validity,
) -> ArrayRef {
    let sliced = values.slice(start..end);
    DecimalArray::new(sliced, decimal_dtype, validity).into_array()
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DecimalDType;

    use crate::Array;
    use crate::arrays::DecimalArray;
    use crate::compute::slice;
    use crate::validity::Validity;

    #[test]
    fn test_slice() {
        let array = DecimalArray::new(
            buffer![100i128, 200i128, 300i128],
            DecimalDType::new(3, 2),
            Validity::NonNullable,
        );

        let sliced = slice(&array, 1, 3).unwrap();
        assert_eq!(sliced.len(), 2);

        let decimal = sliced.as_any().downcast_ref::<DecimalArray>().unwrap();
        assert_eq!(decimal.buffer::<i128>(), buffer![200i128, 300i128]);
    }
}

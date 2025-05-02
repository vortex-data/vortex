use vortex_buffer::Buffer;
use vortex_dtype::DecimalDType;
use vortex_error::VortexResult;

use crate::arrays::{DecimalArray, NativeDecimalType};
use crate::validity::Validity;
use crate::{Array, ArrayOperationsImpl, ArrayRef, match_each_decimal_value_type};

impl ArrayOperationsImpl for DecimalArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        match_each_decimal_value_type!(self.values_type, |$S| {
            slice_typed(
                self.buffer::<$S>(),
                start,
                stop,
                self.decimal_dtype(),
                self.validity.clone(),
            )
        })
    }
}

fn slice_typed<T: NativeDecimalType>(
    values: Buffer<T>,
    start: usize,
    end: usize,
    decimal_dtype: DecimalDType,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    let sliced = values.slice(start..end);
    let validity = validity.slice(start, end)?;
    Ok(DecimalArray::new(sliced, decimal_dtype, validity).into_array())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DecimalDType;

    use crate::Array;
    use crate::arrays::DecimalArray;
    use crate::validity::Validity;

    #[test]
    fn test_slice() {
        let array = DecimalArray::new(
            buffer![100i128, 200i128, 300i128, 4000i128],
            DecimalDType::new(3, 2),
            Validity::NonNullable,
        )
        .to_array();

        let sliced = array.slice(1, 3).unwrap();
        assert_eq!(sliced.len(), 2);

        let decimal = sliced.as_any().downcast_ref::<DecimalArray>().unwrap();
        assert_eq!(decimal.buffer::<i128>(), buffer![200i128, 300i128]);
    }

    #[test]
    fn test_slice_nullable() {
        let array = DecimalArray::new(
            buffer![100i128, 200i128, 300i128, 4000i128],
            DecimalDType::new(3, 2),
            Validity::from_iter([false, true, false, true]),
        )
        .to_array();

        let sliced = array.slice(1, 3).unwrap();
        assert_eq!(sliced.len(), 2);
    }
}

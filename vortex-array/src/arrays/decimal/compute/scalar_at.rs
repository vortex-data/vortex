use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{DecimalArray, DecimalEncoding};
use crate::compute::ScalarAtFn;
use crate::{Array, match_each_decimal_value_type};

impl ScalarAtFn<&DecimalArray> for DecimalEncoding {
    fn scalar_at(&self, array: &DecimalArray, index: usize) -> VortexResult<Scalar> {
        let scalar = match_each_decimal_value_type!(array.values_type(), |($D, $CTor)| {
           Scalar::decimal(
                $CTor(array.buffer::<$D>()[index]),
                array.decimal_dtype(),
                array.dtype().nullability(),
            )
        });
        Ok(scalar)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::{DecimalDType, Nullability};
    use vortex_scalar::{DecimalValue, Scalar};

    use crate::arrays::DecimalArray;
    use crate::compute::scalar_at;
    use crate::validity::Validity;

    #[test]
    fn test_scalar_at() {
        let array = DecimalArray::new(
            buffer![100i128],
            DecimalDType::new(3, 2),
            Validity::NonNullable,
        );

        assert_eq!(
            scalar_at(&array, 0).unwrap(),
            Scalar::decimal(
                DecimalValue::I128(100),
                DecimalDType::new(3, 2),
                Nullability::NonNullable
            )
        );
    }
}

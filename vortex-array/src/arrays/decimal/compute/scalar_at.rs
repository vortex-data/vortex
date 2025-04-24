use vortex_error::VortexResult;
use vortex_scalar::{DecimalValue, Scalar, i256};

use crate::Array;
use crate::arrays::decimal::serde::DecimalValueType;
use crate::arrays::{DecimalArray, DecimalEncoding};
use crate::compute::ScalarAtFn;

impl ScalarAtFn<&DecimalArray> for DecimalEncoding {
    fn scalar_at(&self, array: &DecimalArray, index: usize) -> VortexResult<Scalar> {
        Ok(match array.values_type {
            DecimalValueType::I128 => Scalar::decimal(
                DecimalValue::I128(array.buffer::<i128>()[index]),
                array.decimal_dtype(),
                array.dtype().nullability(),
            ),
            DecimalValueType::I256 => Scalar::decimal(
                DecimalValue::I256(array.buffer::<i256>()[index]),
                array.decimal_dtype(),
                array.dtype().nullability(),
            ),
        })
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

use vortex_error::VortexResult;
use vortex_scalar::i256;

use crate::arrays::decimal::serde::DecimalValueType;
use crate::arrays::{DecimalArray, DecimalEncoding, NativeDecimalType};
use crate::compute::{IsConstantFn, IsConstantOpts};

impl IsConstantFn<&DecimalArray> for DecimalEncoding {
    fn is_constant(
        &self,
        array: &DecimalArray,
        _opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        match array.values_type {
            DecimalValueType::I128 => Ok(Some(compute_is_constant(&array.buffer::<i128>()))),
            DecimalValueType::I256 => Ok(Some(compute_is_constant(&array.buffer::<i256>()))),
        }
    }
}

fn compute_is_constant<T: NativeDecimalType>(values: &[T]) -> bool {
    // We know that the top-level `is_constant` ensures that the array is all_valid or non-null.
    let first_value = values[0];

    for &value in &values[1..] {
        if value != first_value {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DecimalDType;

    use crate::arrays::DecimalArray;
    use crate::compute::is_constant;
    use crate::validity::Validity;

    #[test]
    fn test_is_constant() {
        let array = DecimalArray::new(
            buffer![0i128, 1i128, 2i128],
            DecimalDType::new(19, 0),
            Validity::NonNullable,
        );

        assert!(!is_constant(&array).unwrap());

        let array = DecimalArray::new(
            buffer![100i128, 100i128, 100i128],
            DecimalDType::new(19, 0),
            Validity::NonNullable,
        );

        assert!(is_constant(&array).unwrap());
    }
}

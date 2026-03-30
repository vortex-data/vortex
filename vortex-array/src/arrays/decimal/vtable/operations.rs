// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::Decimal;
use crate::arrays::decimal::vtable::DecimalArray;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<Decimal> for Decimal {
    fn scalar_at(
        array: &DecimalArray,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(match_each_decimal_value_type!(array.values_type(), |D| {
            Scalar::decimal(
                DecimalValue::from(array.buffer::<D>()[index]),
                array.decimal_dtype(),
                array.dtype().nullability(),
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::DynArray;
    use crate::IntoArray;
    use crate::arrays::Decimal;
    use crate::arrays::DecimalArray;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn test_slice() {
        let array = DecimalArray::new(
            buffer![100i128, 200i128, 300i128, 4000i128],
            DecimalDType::new(3, 2),
            Validity::NonNullable,
        )
        .into_array();

        let sliced = array.slice(1..3).unwrap();
        assert_eq!(sliced.len(), 2);

        let decimal = sliced.as_::<Decimal>();
        assert_eq!(decimal.buffer::<i128>(), buffer![200i128, 300i128]);
    }

    #[test]
    fn test_slice_nullable() {
        let array = DecimalArray::new(
            buffer![100i128, 200i128, 300i128, 4000i128],
            DecimalDType::new(3, 2),
            Validity::from_iter([false, true, false, true]),
        )
        .into_array();

        let sliced = array.slice(1..3).unwrap();
        assert_eq!(sliced.len(), 2);
    }

    #[test]
    fn test_scalar_at() {
        let array = DecimalArray::new(
            buffer![100i128],
            DecimalDType::new(3, 2),
            Validity::NonNullable,
        );

        assert_eq!(
            array.scalar_at(0).unwrap(),
            Scalar::decimal(
                DecimalValue::I128(100),
                DecimalDType::new(3, 2),
                Nullability::NonNullable
            )
        );
    }
}

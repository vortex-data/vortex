// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Decimal;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::DecimalArraySlotsExt;
use crate::integer;
use crate::scalar::Scalar;

impl OperationsVTable<Decimal> for Decimal {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Decimal>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let value = array.values().execute_scalar(index, ctx)?;
        Ok(Scalar::decimal(
            integer::scalar_value(&value)?,
            array.decimal_dtype(),
            array.dtype().nullability(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::TEST_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::Decimal;
    use crate::arrays::DecimalArray;
    use crate::arrays::decimal::DecimalArrayExt;
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

        let decimal = sliced
            .as_::<Decimal>()
            .materialize_values(&mut TEST_SESSION.create_execution_ctx())
            .unwrap();
        assert_eq!(decimal.buffer::<i16>(), buffer![200i16, 300i16]);
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
            array
                .execute_scalar(0, &mut TEST_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::decimal(
                DecimalValue::I128(100),
                DecimalDType::new(3, 2),
                Nullability::NonNullable
            )
        );
    }
}

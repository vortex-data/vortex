// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Decimal;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;

impl OperationsVTable<Decimal> for Decimal {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Decimal>,
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
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Decimal;
    use crate::arrays::DecimalArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_slice() {
        let array = DecimalArray::new(
            buffer![100i16, 200i16, 300i16, 4000i16],
            DecimalDType::new(4, 2),
            Validity::NonNullable,
        )
        .into_array();

        let sliced = array.slice(1..3).unwrap();
        assert_eq!(sliced.len(), 2);

        let decimal = sliced.as_::<Decimal>();
        assert_eq!(decimal.buffer::<i16>(), buffer![200i16, 300i16]);
    }

    #[test]
    fn test_slice_nullable() {
        let array = DecimalArray::new(
            buffer![100i16, 200i16, 300i16, 4000i16],
            DecimalDType::new(4, 2),
            Validity::from_iter([false, true, false, true]),
        )
        .into_array();

        let sliced = array.slice(1..3).unwrap();
        assert_eq!(sliced.len(), 2);
        assert_arrays_eq!(
            sliced,
            DecimalArray::from_option_iter([Some(200i16), None], DecimalDType::new(4, 2)),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn test_scalar_at() {
        let array = DecimalArray::new(
            buffer![100i16],
            DecimalDType::new(3, 2),
            Validity::NonNullable,
        );

        assert_eq!(
            array
                .execute_scalar(0, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::decimal(
                DecimalValue::I128(100),
                DecimalDType::new(3, 2),
                Nullability::NonNullable
            )
        );
    }
}

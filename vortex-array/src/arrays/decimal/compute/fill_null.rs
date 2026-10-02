// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::DecimalArraySlotsExt;
use crate::integer;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullKernel;

impl FillNullKernel for Decimal {
    fn fill_null(
        array: ArrayView<'_, Decimal>,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let value = fill_value
            .as_decimal()
            .decimal_value()
            .vortex_expect("Non-null decimal fill value");
        let dtype = array
            .values_dtype()
            .with_nullability(fill_value.dtype().nullability());
        let value = integer::scalar_from_integer(value, &dtype)?;
        let values = integer::fill_null(array.values(), &value, ctx)?;
        Ok(Some(
            DecimalArray::try_new_values(values, array.decimal_dtype())?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::DecimalArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn fill_null_leading_none() {
        let mut ctx = array_session().create_execution_ctx();
        let decimal_dtype = DecimalDType::new(19, 2);
        let arr = DecimalArray::from_option_iter(
            [None, Some(800i128), None, Some(1000i128), None],
            decimal_dtype,
        );
        let p = arr
            .into_array()
            .fill_null(Scalar::decimal(
                DecimalValue::I128(4200i128),
                DecimalDType::new(19, 2),
                Nullability::NonNullable,
            ))
            .unwrap()
            .execute::<DecimalArray>(&mut ctx)
            .unwrap()
            .materialize_values(&mut ctx)
            .unwrap();
        assert_arrays_eq!(
            p,
            DecimalArray::from_iter([4200, 800, 4200, 1000, 4200], decimal_dtype),
            &mut ctx
        );
        assert_eq!(
            p.buffer::<i128>().as_slice(),
            vec![4200, 800, 4200, 1000, 4200]
        );
        assert!(
            p.as_ref()
                .validity()
                .unwrap()
                .execute_mask(
                    p.as_ref().len(),
                    &mut array_session().create_execution_ctx()
                )
                .unwrap()
                .all_true()
        );
    }

    #[test]
    fn fill_null_all_none() {
        let mut ctx = array_session().create_execution_ctx();
        let decimal_dtype = DecimalDType::new(19, 2);

        let arr = DecimalArray::from_option_iter(
            [Option::<i128>::None, None, None, None, None],
            decimal_dtype,
        );

        let p = arr
            .into_array()
            .fill_null(Scalar::decimal(
                DecimalValue::I128(25500i128),
                DecimalDType::new(19, 2),
                Nullability::NonNullable,
            ))
            .unwrap()
            .execute::<DecimalArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(
            p,
            DecimalArray::from_iter([25500, 25500, 25500, 25500, 25500], decimal_dtype),
            &mut ctx
        );
    }

    /// fill_null with a value that overflows the array's storage type should upcast the array.
    #[test]
    fn fill_null_overflow_upcasts() {
        let mut ctx = array_session().create_execution_ctx();
        let decimal_dtype = DecimalDType::new(3, 0);
        let arr = DecimalArray::from_option_iter([None, Some(10i8), None], decimal_dtype);
        // The fill value needs i16 storage because 200 exceeds i8::MAX.
        let result = arr
            .into_array()
            .fill_null(Scalar::decimal(
                DecimalValue::I128(200i128),
                DecimalDType::new(3, 0),
                Nullability::NonNullable,
            ))
            .unwrap()
            .execute::<DecimalArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(
            result,
            DecimalArray::from_iter([200i16, 10, 200], decimal_dtype),
            &mut ctx
        );
    }

    #[test]
    fn fill_null_non_nullable() {
        let mut ctx = array_session().create_execution_ctx();
        let decimal_dtype = DecimalDType::new(19, 2);

        let arr = DecimalArray::new(
            buffer![800i128, 1000i128, 1200i128, 1400i128, 1600i128],
            decimal_dtype,
            Validity::NonNullable,
        );
        let p = arr
            .into_array()
            .fill_null(Scalar::decimal(
                DecimalValue::I128(25500i128),
                DecimalDType::new(19, 2),
                Nullability::NonNullable,
            ))
            .unwrap()
            .execute::<DecimalArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(
            p,
            DecimalArray::from_iter([800i128, 1000, 1200, 1400, 1600], decimal_dtype),
            &mut ctx
        );
    }
}

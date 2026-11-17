// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullKernel;
use crate::validity::Validity;

impl FillNullKernel for Decimal {
    fn fill_null(
        array: ArrayView<'_, Decimal>,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Validity::Array(is_valid) = array.validity()? else {
            unreachable!("checked in entry point");
        };
        let is_invalid = is_valid.execute::<BoolArray>(ctx)?.into_bit_buffer().not();
        let fill = fill_value
            .as_decimal()
            .decimal_value()
            .vortex_expect("fill_null requires a non-null fill value");
        let validity = Validity::from(fill_value.dtype().nullability());

        let result = match_each_decimal_value_type!(array.values_type(), |T| {
            fill_buffer::<T>(array, &is_invalid, &fill, validity)
        });
        Ok(Some(result.into_array()))
    }
}

/// Replaces the value of each row set in `is_invalid` with `fill`, cast to the canonical type `T`
/// of the array.
fn fill_buffer<T: NativeDecimalType>(
    array: ArrayView<'_, Decimal>,
    is_invalid: &BitBuffer,
    fill: &DecimalValue,
    validity: Validity,
) -> DecimalArray {
    let fill = fill
        .cast::<T>()
        .vortex_expect("fill scalar fits the array's decimal precision");
    let mut buffer = array.buffer::<T>().into_mut();
    buffer.fill_where::<false>(is_invalid, fill);

    // SAFETY: Newly valid slots contain the fill scalar. Other values keep their original
    // precision, and the fill preserves the length.
    unsafe { DecimalArray::new_unchecked(buffer.freeze(), array.decimal_dtype(), validity) }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::DecimalArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DecimalDType;
    use crate::dtype::DecimalType;
    use crate::dtype::Nullability;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn fill_null_leading_none() {
        let mut ctx = SESSION.create_execution_ctx();
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
                .execute_mask(p.as_ref().len(), &mut SESSION.create_execution_ctx())
                .unwrap()
                .all_true()
        );
    }

    #[test]
    fn fill_null_all_none() {
        let mut ctx = SESSION.create_execution_ctx();
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
            .unwrap();
        assert_arrays_eq!(
            p,
            DecimalArray::from_iter([25500, 25500, 25500, 25500, 25500], decimal_dtype),
            &mut ctx
        );
    }

    #[rstest]
    #[case::canonical(DecimalValue::I16(200))]
    #[case::wide(DecimalValue::I128(200))]
    fn fill_null_preserves_canonical_width(#[case] fill: DecimalValue) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DecimalDType::new(3, 0);
        let result = DecimalArray::from_option_iter([None, Some(10i8), None], dtype)
            .into_array()
            .fill_null(Scalar::decimal(fill, dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(result.values_type(), DecimalType::I16);
        assert_arrays_eq!(
            result,
            DecimalArray::from_iter([200i16, 10, 200], dtype),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn fill_null_non_nullable() {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(19, 2);

        let arr = DecimalArray::from_iter(
            [800i128, 1000i128, 1200i128, 1400i128, 1600i128],
            decimal_dtype,
        );
        let p = arr
            .into_array()
            .fill_null(Scalar::decimal(
                DecimalValue::I128(25500i128),
                DecimalDType::new(19, 2),
                Nullability::NonNullable,
            ))
            .unwrap();
        assert_arrays_eq!(
            p,
            DecimalArray::from_iter([800i128, 1000, 1200, 1400, 1600], decimal_dtype),
            &mut ctx
        );
    }
}

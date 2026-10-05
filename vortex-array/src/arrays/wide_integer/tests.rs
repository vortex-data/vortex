// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::WideIntegerArray;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_sorted::is_sorted;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::Narrow;
use crate::arrays::NarrowArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::narrow::NarrowArraySlotsExt;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::dtype::integer::integer_dtype;
use crate::integer;
use crate::scalar::DecimalValue;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

#[rstest]
#[case::i128(DecimalType::I128)]
#[case::i256(DecimalType::I256)]
fn test_signed_order_and_full_integer_range(#[case] width: DecimalType) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values = if width == DecimalType::I128 {
        WideIntegerArray::try_new(
            buffer![i128::MIN, -256, -1, 0, 255, 256, i128::MAX],
            Validity::NonNullable,
        )?
        .into_array()
    } else {
        WideIntegerArray::try_new(
            buffer![
                i256::MIN,
                i256::from_i128(-256),
                i256::from_i128(-1),
                i256::ZERO,
                i256::from_i128(255),
                i256::from_i128(256),
                i256::MAX,
            ],
            Validity::NonNullable,
        )?
        .into_array()
    };
    let reversed = values.take(buffer![6u32, 5, 4, 3, 2, 1, 0].into_array())?;
    assert_arrays_eq!(
        values.binary(reversed, Operator::Lt)?,
        BoolArray::from_iter([true, true, true, false, false, false, false]),
        &mut ctx
    );
    assert!(is_sorted(&values, &mut ctx)?);
    let bounds = min_max(&values, &mut ctx, NumericalAggregateOpts::default())?
        .vortex_expect("Non-empty integer bounds");
    assert!(bounds.min < bounds.max);
    assert_eq!(bounds.min, values.execute_scalar(0, &mut ctx)?);
    assert_eq!(bounds.max, values.execute_scalar(6, &mut ctx)?);

    Ok(())
}

#[test]
fn test_narrowing_ignores_invalid_wide_payloads() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values = WideIntegerArray::try_new(
        buffer![127i128, i128::MIN, -128],
        Validity::from_iter([true, false, true]),
    )?
    .into_array();
    let narrow = NarrowArray::encode_signed(values.clone(), &mut ctx)?;
    assert_eq!(
        narrow.dtype(),
        &integer_dtype(DecimalType::I128, Nullability::Nullable)
    );
    assert_eq!(
        narrow.as_::<Narrow>().values().dtype(),
        &DType::Primitive(PType::I8, Nullability::Nullable)
    );
    assert_arrays_eq!(narrow, values, &mut ctx);

    Ok(())
}

#[test]
fn test_integer_cast_checks_valid_lanes() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values = WideIntegerArray::try_new(
        buffer![127i128, i128::MIN],
        Validity::from_iter([true, false]),
    )?
    .into_array();
    let dtype = DType::Primitive(PType::I8, Nullability::Nullable);
    assert_arrays_eq!(
        values.cast(dtype.clone())?,
        PrimitiveArray::from_option_iter([Some(127i8), None]),
        &mut ctx
    );
    assert!(
        WideIntegerArray::try_new(buffer![128i128], Validity::NonNullable)?
            .into_array()
            .cast(PType::I8.into())?
            .execute::<PrimitiveArray>(&mut ctx)
            .is_err()
    );
    assert!(integer::scalar_from_integer(DecimalValue::I128(128), &dtype).is_err());

    Ok(())
}

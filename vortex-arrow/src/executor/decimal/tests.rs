// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal export equivalence and width checks for canonical arrays and `Widen` children.

use std::sync::LazyLock;

use arrow_array::Array;
use arrow_array::Decimal64Array;
use arrow_array::Decimal128Array;
use arrow_array::Decimal256Array;
use arrow_array::StructArray as ArrowStructArray;
use arrow_buffer::i256;
use arrow_schema::DataType;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::WidenArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::DecimalBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrowArrayExecutor;
use crate::executor::decimal::DecimalArray;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// Presents `values` as a `Widen` with the logical `precision`, and the scale and nullability
/// of `values`.
fn widen(values: DecimalArray, precision: u8) -> VortexResult<ArrayRef> {
    let decimal_dtype = DecimalDType::new(precision, values.decimal_dtype().scale());
    let dtype = DType::Decimal(decimal_dtype, values.dtype().nullability());
    Ok(WidenArray::try_new(values.into_array(), dtype)?.into_array())
}

/// A child at the largest precision that `T` holds, with the smallest and largest values of
/// that precision in its valid rows.
///
/// A nullable child holds `garbage` in its null rows. Arrow equality ignores null rows, so the
/// export must match the canonical export whatever the null rows hold.
fn widen_child<T: NativeDecimalType>(garbage: T, nullable: bool) -> DecimalArray {
    let decimal_dtype = DecimalDType::new(T::MAX_PRECISION, 1);
    let min = T::MIN_BY_PRECISION[usize::from(T::MAX_PRECISION)];
    let max = T::MAX_BY_PRECISION[usize::from(T::MAX_PRECISION)];

    if nullable {
        DecimalArray::new(
            Buffer::from_iter([min, garbage, max, min, garbage]),
            decimal_dtype,
            Validity::from_iter([true, false, true, true, false]),
        )
    } else {
        DecimalArray::new(
            Buffer::from_iter([min, max, max, min, max]),
            decimal_dtype,
            Validity::NonNullable,
        )
    }
}

#[test]
fn decimal_to_arrow() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Make a very simple i128 and i256 array.
    let decimal_vortex = DecimalArray::new(
        buffer![1i128, 2i128, 3i128, 4i128, 5i128],
        DecimalDType::new(19, 2),
        Validity::NonNullable,
    );
    let arrow = decimal_vortex
        .into_array()
        .execute_arrow(Some(&DataType::Decimal128(19, 2)), &mut ctx)?;
    assert_eq!(arrow.data_type(), &DataType::Decimal128(19, 2));
    let decimal_array = arrow.as_any().downcast_ref::<Decimal128Array>().unwrap();
    assert_eq!(
        decimal_array.values().as_ref(),
        &[1i128, 2i128, 3i128, 4i128, 5i128]
    );
    assert_eq!(
        decimal_array,
        &Decimal128Array::from(vec![1, 2, 3, 4, 5]).with_precision_and_scale(19, 2)?
    );
    Ok(())
}

#[rstest]
#[case(0i8)]
#[case(0i16)]
#[case(0i32)]
#[case(0i64)]
#[case(0i128)]
#[case(vortex_array::dtype::i256::ZERO)]
fn test_to_arrow_decimal128<T: NativeDecimalType>(#[case] _decimal_type: T) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut decimal = DecimalBuilder::new_in::<T>(
        DecimalDType::new(2, 1),
        false.into(),
        vortex_buffer::BufferAllocatorRef::static_ref(),
    );
    decimal.append_value(10);
    decimal.append_value(11);
    decimal.append_value(12);
    let decimal = decimal.finish();

    let arrow_array = decimal.execute_arrow(Some(&DataType::Decimal128(2, 1)), &mut ctx)?;
    let arrow_decimal = arrow_array
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    assert_eq!(arrow_decimal.value(0), 10);
    assert_eq!(arrow_decimal.value(1), 11);
    assert_eq!(arrow_decimal.value(2), 12);
    assert_eq!(
        arrow_decimal,
        &Decimal128Array::from(vec![10, 11, 12]).with_precision_and_scale(2, 1)?
    );
    Ok(())
}

#[rstest]
#[case(0i8)]
#[case(0i16)]
#[case(0i32)]
#[case(0i64)]
#[case(0i128)]
#[case(vortex_array::dtype::i256::ZERO)]
fn test_to_arrow_decimal32<T: NativeDecimalType>(#[case] _decimal_type: T) -> VortexResult<()> {
    use arrow_array::Decimal32Array;

    let mut ctx = SESSION.create_execution_ctx();
    let mut decimal = DecimalBuilder::new_in::<T>(
        DecimalDType::new(2, 1),
        false.into(),
        vortex_buffer::BufferAllocatorRef::static_ref(),
    );
    decimal.append_value(10);
    decimal.append_value(11);
    decimal.append_value(12);
    let decimal = decimal.finish();

    let arrow_array = decimal.execute_arrow(Some(&DataType::Decimal32(2, 1)), &mut ctx)?;
    let arrow_decimal = arrow_array
        .as_any()
        .downcast_ref::<Decimal32Array>()
        .unwrap();
    assert_eq!(arrow_decimal.value(0), 10);
    assert_eq!(arrow_decimal.value(1), 11);
    assert_eq!(arrow_decimal.value(2), 12);
    assert_eq!(
        arrow_decimal,
        &Decimal32Array::from(vec![10, 11, 12]).with_precision_and_scale(2, 1)?
    );
    Ok(())
}

#[rstest]
#[case(0i8)]
#[case(0i16)]
#[case(0i32)]
#[case(0i64)]
#[case(0i128)]
#[case(vortex_array::dtype::i256::ZERO)]
fn test_to_arrow_decimal64<T: NativeDecimalType>(#[case] _decimal_type: T) -> VortexResult<()> {
    use arrow_array::Decimal64Array;

    let mut ctx = SESSION.create_execution_ctx();
    let mut decimal = DecimalBuilder::new_in::<T>(
        DecimalDType::new(2, 1),
        false.into(),
        vortex_buffer::BufferAllocatorRef::static_ref(),
    );
    decimal.append_value(10);
    decimal.append_value(11);
    decimal.append_value(12);
    let decimal = decimal.finish();

    let arrow_array = decimal.execute_arrow(Some(&DataType::Decimal64(2, 1)), &mut ctx)?;
    let arrow_decimal = arrow_array
        .as_any()
        .downcast_ref::<Decimal64Array>()
        .unwrap();
    assert_eq!(arrow_decimal.value(0), 10);
    assert_eq!(arrow_decimal.value(1), 11);
    assert_eq!(arrow_decimal.value(2), 12);
    assert_eq!(
        arrow_decimal,
        &Decimal64Array::from(vec![10, 11, 12]).with_precision_and_scale(2, 1)?
    );
    Ok(())
}

#[rstest]
#[case(0i8)]
#[case(0i16)]
#[case(0i32)]
#[case(0i64)]
#[case(0i128)]
#[case(vortex_array::dtype::i256::ZERO)]
fn test_to_arrow_decimal256<T: NativeDecimalType>(#[case] _decimal_type: T) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut decimal = DecimalBuilder::new_in::<T>(
        DecimalDType::new(2, 1),
        false.into(),
        vortex_buffer::BufferAllocatorRef::static_ref(),
    );
    decimal.append_value(10);
    decimal.append_value(11);
    decimal.append_value(12);
    let decimal = decimal.finish();

    let arrow_array = decimal.execute_arrow(Some(&DataType::Decimal256(2, 1)), &mut ctx)?;
    let arrow_decimal = arrow_array
        .as_any()
        .downcast_ref::<Decimal256Array>()
        .unwrap();
    assert_eq!(arrow_decimal.value(0), i256::from_i128(10));
    assert_eq!(arrow_decimal.value(1), i256::from_i128(11));
    assert_eq!(arrow_decimal.value(2), i256::from_i128(12));
    assert_eq!(
        arrow_decimal,
        &Decimal256Array::from(vec![
            i256::from_i128(10),
            i256::from_i128(11),
            i256::from_i128(12)
        ])
        .with_precision_and_scale(2, 1)?
    );
    Ok(())
}

#[rstest]
#[case::i8_p4_decimal32(i8::MAX, 4, DataType::Decimal32(4, 1))]
#[case::i8_p4_decimal64(i8::MAX, 4, DataType::Decimal64(4, 1))]
#[case::i8_p4_decimal128(i8::MAX, 4, DataType::Decimal128(4, 1))]
#[case::i8_p4_decimal256(i8::MAX, 4, DataType::Decimal256(4, 1))]
#[case::i32_p18_decimal64(i32::MAX, 18, DataType::Decimal64(18, 1))]
#[case::i32_p18_decimal128(i32::MAX, 18, DataType::Decimal128(18, 1))]
#[case::i32_p18_decimal256(i32::MAX, 18, DataType::Decimal256(18, 1))]
#[case::i64_p38_decimal128(i64::MAX, 38, DataType::Decimal128(38, 1))]
#[case::i64_p38_decimal256(i64::MAX, 38, DataType::Decimal256(38, 1))]
#[case::i128_p76_decimal256(i128::MAX, 76, DataType::Decimal256(76, 1))]
fn test_widen_to_arrow_matches_canonical<T: NativeDecimalType>(
    #[case] garbage: T,
    #[case] precision: u8,
    #[case] data_type: DataType,
    #[values(false, true)] nullable: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = widen(widen_child(garbage, nullable), precision)?;
    let canonical = array
        .clone()
        .execute::<DecimalArray>(&mut ctx)?
        .into_array();

    let actual = array.execute_arrow(Some(&data_type), &mut ctx)?;
    let expected = canonical.execute_arrow(Some(&data_type), &mut ctx)?;
    assert_eq!(actual.data_type(), &data_type);
    assert_eq!(&actual, &expected);

    Ok(())
}

#[test]
fn test_widen_to_arrow_uses_logical_precision() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = DecimalArray::new(
        buffer![-99i8, 0, 99],
        DecimalDType::new(2, 1),
        Validity::NonNullable,
    );

    let arrow = widen(values, 4)?.execute_arrow(None, &mut ctx)?;
    assert_eq!(
        arrow.as_any().downcast_ref::<Decimal128Array>().unwrap(),
        &Decimal128Array::from(vec![-99, 0, 99]).with_precision_and_scale(4, 1)?
    );

    Ok(())
}

#[test]
fn test_sliced_widen_to_arrow() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = DecimalArray::new(
        buffer![10i32, i32::MAX, 30, 40, 50],
        DecimalDType::new(9, 2),
        Validity::from_iter([true, false, true, true, true]),
    );

    let sliced = widen(values, 18)?.slice(1..4)?;
    let arrow = sliced.execute_arrow(Some(&DataType::Decimal128(18, 2)), &mut ctx)?;
    assert_eq!(
        arrow.as_any().downcast_ref::<Decimal128Array>().unwrap(),
        &Decimal128Array::from(vec![None, Some(30), Some(40)]).with_precision_and_scale(18, 2)?
    );

    Ok(())
}

#[test]
fn test_struct_with_widen_field_to_arrow() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = DecimalArray::new(
        buffer![1i32, 2, 3],
        DecimalDType::new(9, 2),
        Validity::NonNullable,
    );
    let array = StructArray::from_fields(&[("price", widen(values, 18)?)])?.into_array();

    let arrow = array.execute_arrow(None, &mut ctx)?;
    let arrow = arrow.as_any().downcast_ref::<ArrowStructArray>().unwrap();
    assert_eq!(
        arrow
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap(),
        &Decimal128Array::from(vec![1, 2, 3]).with_precision_and_scale(18, 2)?
    );

    Ok(())
}

/// A child stored wider than its canonical width, with a value that does not fit in the Arrow
/// width in a null row. Some or all of the rows are null. The export narrows the child, so it
/// must skip the null rows.
#[rstest]
#[case::partially_null(Validity::from_iter([true, false]), vec![Some(1), None])]
#[case::all_null(Validity::AllInvalid, vec![None, None])]
fn test_widen_to_arrow_skips_null_rows_when_narrowing(
    #[case] validity: Validity,
    #[case] expected: Vec<Option<i64>>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = DecimalArray::new(buffer![1i128, i128::MAX], DecimalDType::new(9, 2), validity);

    let arrow = widen(values, 18)?.execute_arrow(Some(&DataType::Decimal64(18, 2)), &mut ctx)?;
    assert_eq!(
        arrow.as_any().downcast_ref::<Decimal64Array>().unwrap(),
        &Decimal64Array::from(expected).with_precision_and_scale(18, 2)?
    );

    Ok(())
}

#[rstest]
#[case::all_valid(Validity::NonNullable)]
#[case::partially_valid(Validity::from_iter([true, false]))]
fn test_to_arrow_rejects_valid_value_that_does_not_fit(
    #[case] validity: Validity,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = DecimalArray::new(buffer![i128::MAX, 0], DecimalDType::new(9, 2), validity);

    let err = values
        .into_array()
        .execute_arrow(Some(&DataType::Decimal64(9, 2)), &mut ctx)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("every valid i128 value must fit in i64 to narrow")
    );

    Ok(())
}

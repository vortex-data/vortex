// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal casts preserve lazy widening, checked precision and nullability.
//!
//! These cases also cover decimal buffers whose stored width differs from their canonical width.

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::SESSION;
use super::canonical;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::DictArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::validity::Validity;

#[rstest]
#[case::i8(DecimalArray::from_iter([12i8, -12], DecimalDType::new(2, 0)), 4, vec![12, -12])]
#[case::i16(DecimalArray::from_iter([12i16, -12], DecimalDType::new(4, 0)), 9, vec![12, -12])]
#[case::i32(DecimalArray::from_iter([12i32, -12], DecimalDType::new(9, 0)), 18, vec![12, -12])]
#[case::i64(DecimalArray::from_iter([12i64, -12], DecimalDType::new(18, 0)), 38, vec![12, -12])]
#[case::i128(DecimalArray::from_iter([12i128, -12], DecimalDType::new(38, 0)), 76, vec![12, -12])]
#[case::stored_target_width(
    DecimalArray::from_iter([12i16, -12], DecimalDType::new(2, 0)),
    4,
    vec![12, -12],
)]
#[case::stored_wider_than_target(
    DecimalArray::from_iter([12i128, -12], DecimalDType::new(2, 0)),
    4,
    vec![12, -12],
)]
#[case::stored_i256(
    DecimalArray::from_iter(
        [i256::from_i128(12), i256::from_i128(-12)],
        DecimalDType::new(2, 0),
    ),
    4,
    vec![12, -12],
)]
#[case::empty(DecimalArray::from_iter(Vec::<i8>::new(), DecimalDType::new(2, 0)), 4, vec![])]
#[case::all_null(
    DecimalArray::new(buffer![12i8, -12], DecimalDType::new(2, 0), Validity::AllInvalid),
    4,
    vec![0, 0],
)]
fn test_precision_growth_is_lazy(
    #[case] array: DecimalArray,
    #[case] precision: u8,
    #[case] expected: Vec<i128>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let source_dtype = array.dtype().clone();
    let source_values_type = array.values_type();
    let validity = array.validity()?;
    let decimal_dtype = DecimalDType::new(precision, 0);
    let target = DType::Decimal(decimal_dtype, validity.nullability());

    let casted = array.into_array().cast(target.clone())?;

    assert!(casted.is::<Widen>());
    let widen = casted.as_::<Widen>();
    let values = widen.values();
    assert_eq!(values.dtype(), &source_dtype);
    assert_eq!(values.as_::<Decimal>().values_type(), source_values_type);
    let executed = casted.execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(executed.dtype(), &target);
    assert_eq!(
        executed.values_type(),
        decimal_dtype.canonical_values_type()
    );
    let expected = DecimalArray::new(Buffer::from_iter(expected), decimal_dtype, validity);
    assert_arrays_eq!(executed, expected, &mut ctx);

    Ok(())
}

#[rstest]
#[case::to_nullable(Validity::NonNullable, Nullability::Nullable)]
#[case::to_non_nullable(Validity::AllValid, Nullability::NonNullable)]
#[case::keeps_nullable(Validity::AllValid, Nullability::Nullable)]
fn test_precision_growth_sets_child_nullability(
    #[case] validity: Validity,
    #[case] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = DecimalArray::new(buffer![12i8, -12], DecimalDType::new(2, 0), validity);
    let target_decimal = DecimalDType::new(4, 0);
    let target = DType::Decimal(target_decimal, nullability);

    let casted = array.into_array().cast(target.clone())?;

    assert!(casted.is::<Widen>());
    assert_eq!(
        casted.as_::<Widen>().values().dtype().nullability(),
        nullability
    );
    assert_eq!(casted.dtype(), &target);
    let expected_validity = match nullability {
        Nullability::Nullable => Validity::AllValid,
        Nullability::NonNullable => Validity::NonNullable,
    };
    let expected = DecimalArray::new(buffer![12i16, -12], target_decimal, expected_validity);
    assert_arrays_eq!(casted, expected, &mut ctx);

    Ok(())
}

/// The child holds the same values as the source, so both cast paths keep its exact extrema, even
/// when the cast changes the nullability.
#[rstest]
#[case::reduce(false, Nullability::NonNullable)]
#[case::kernel(true, Nullability::NonNullable)]
#[case::kernel_to_nullable(true, Nullability::Nullable)]
fn test_precision_growth_keeps_values_statistics(
    #[case] through_kernel: bool,
    #[case] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let source_decimal = DecimalDType::new(4, 0);
    let extremum = |value: i16, nullability| {
        Scalar::decimal(DecimalValue::I16(value), source_decimal, nullability)
    };
    let array = DecimalArray::from_iter([12i16, -12], source_decimal);

    for (stat, value) in [(Stat::Min, -12), (Stat::Max, 12)] {
        let value = extremum(value, Nullability::NonNullable).into_value();
        let value = value.vortex_expect("the planted extremum is not null");
        array.statistics().set(stat, Precision::Exact(value));
    }

    let target = DType::Decimal(DecimalDType::new(18, 0), nullability);
    let casted = if through_kernel {
        <Decimal as CastKernel>::cast(array.as_view(), &target, &mut ctx)?
    } else {
        <Decimal as CastReduce>::cast(array.as_view(), &target)?
    };
    let casted = casted.vortex_expect("same-scale precision growth casts");

    assert!(casted.is::<Widen>());
    let values = casted.as_::<Widen>().values().clone();
    let statistics = values.statistics();
    let min = statistics.get(Stat::Min);
    let max = statistics.get(Stat::Max);
    assert_eq!(min, Precision::Exact(extremum(-12, nullability)));
    assert_eq!(max, Precision::Exact(extremum(12, nullability)));

    Ok(())
}

#[rstest]
#[case::all_valid(true)]
#[case::null_row(false)]
fn test_encoded_validity_falls_back_to_kernel(#[case] second_valid: bool) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let validity = DictArray::try_new(
        buffer![0u8, 1].into_array(),
        BoolArray::from_iter([true, second_valid]).into_array(),
    )?;
    let array = DecimalArray::new(
        buffer![12i8, -12],
        DecimalDType::new(2, 0),
        Validity::Array(validity.into_array()),
    );
    let target_decimal = DecimalDType::new(4, 0);
    let target = DType::Decimal(target_decimal, Nullability::NonNullable);

    assert!(<Decimal as CastReduce>::cast(array.as_view(), &target)?.is_none());
    let result = <Decimal as CastKernel>::cast(array.as_view(), &target, &mut ctx);
    if !second_valid {
        let error = result.expect_err("the encoded validity contains a null row");
        assert!(error.to_string().contains("invalid values to non-nullable"));
        return Ok(());
    }

    let casted = result?.expect("same-scale precision growth has a cast kernel");
    assert!(casted.is::<Widen>());
    assert_eq!(
        casted.as_::<Widen>().values().dtype().nullability(),
        Nullability::NonNullable
    );
    let expected = DecimalArray::from_iter([12i16, -12], target_decimal);
    assert_arrays_eq!(casted, expected, &mut ctx);

    Ok(())
}

#[test]
fn test_same_canonical_width_reuses_stored_buffer() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = DecimalArray::from_iter([12i128, -12], DecimalDType::new(10, 0));
    let source_ptr = array.buffer::<i128>().as_ptr();
    let target_decimal = DecimalDType::new(18, 0);
    let target = DType::Decimal(target_decimal, Nullability::NonNullable);

    let casted = array
        .into_array()
        .cast(target)?
        .execute::<DecimalArray>(&mut ctx)?;

    assert_eq!(casted.buffer::<i128>().as_ptr(), source_ptr);
    assert_arrays_eq!(
        casted,
        DecimalArray::from_iter([12i128, -12], target_decimal),
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case::smaller_decimal(DType::Decimal(DecimalDType::new(3, 2), Nullability::Nullable))]
#[case::wider_decimal(DType::Decimal(DecimalDType::new(38, 2), Nullability::Nullable))]
#[case::rescale(DType::Decimal(DecimalDType::new(18, 3), Nullability::Nullable))]
#[case::integer(DType::Primitive(PType::I64, Nullability::Nullable))]
#[case::float(DType::Primitive(PType::F64, Nullability::Nullable))]
fn test_decimal_widen_cast_matches_canonical(#[case] target: DType) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values =
        DecimalArray::from_option_iter([Some(125i16), None, Some(-250)], DecimalDType::new(4, 2));
    let array = WidenArray::try_new(
        values.into_array(),
        DType::Decimal(DecimalDType::new(18, 2), Nullability::Nullable),
    )?
    .into_array();
    let expected = canonical(&array, &mut ctx)?.cast(target.clone())?;

    let casted = array.cast(target)?;

    assert_arrays_eq!(casted, expected, &mut ctx);

    Ok(())
}

#[test]
fn test_precision_narrowing_checks_values() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = DecimalArray::from_iter([1000i16], DecimalDType::new(4, 0));
    let array = WidenArray::try_new(
        values.into_array(),
        DType::Decimal(DecimalDType::new(18, 0), Nullability::NonNullable),
    )?
    .into_array();
    let target = DType::Decimal(DecimalDType::new(3, 0), Nullability::NonNullable);

    let error = array
        .cast(target)?
        .execute::<DecimalArray>(&mut ctx)
        .expect_err("1000 does not fit a precision of three digits");

    assert!(error.to_string().contains("does not fit in precision"));

    Ok(())
}

#[test]
fn test_repeated_casts_terminate_without_nested_widen() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let source = DecimalArray::from_iter([12i8, -12], DecimalDType::new(2, 0)).into_array();
    let mut array = source.clone();
    for precision in [4, 9, 18, 38, 76, 4, 2] {
        let decimal_dtype = DecimalDType::new(precision, 0);
        let target = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        array = array.cast(target.clone())?;
        assert!(ArrayRef::ptr_eq(&array.cast(target)?, &array));
        if let Some(widen) = array.as_opt::<Widen>() {
            assert!(!widen.values().is::<Widen>());
            assert_eq!(widen.values().dtype(), source.dtype());
        }
        assert_arrays_eq!(
            array.clone(),
            DecimalArray::from_iter([12i128, -12], decimal_dtype),
            &mut ctx
        );
    }

    assert_arrays_eq!(array, source, &mut ctx);

    Ok(())
}

#[test]
fn test_same_canonical_width_copies_narrow_stored_values() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = DecimalArray::from_iter([125i32, -250], DecimalDType::new(10, 2));
    let target_decimal = DecimalDType::new(18, 2);
    let target = DType::Decimal(target_decimal, Nullability::NonNullable);
    let casted = array.into_array().cast(target)?;
    assert!(!casted.is::<Widen>());

    let executed = casted.execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(
        executed.values_type(),
        target_decimal.canonical_values_type()
    );
    assert_arrays_eq!(
        executed,
        DecimalArray::from_iter([125i64, -250], target_decimal),
        &mut ctx
    );

    Ok(())
}

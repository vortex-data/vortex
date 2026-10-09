// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical decimals retain encoded integer children.
//!
//! These cases cover precision-derived child dtypes, materialization, and operations that change
//! stored width. Legacy serialization remains compatible with the buffer representation.

use prost::Message;
use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::Decimal;
use super::DecimalArray;
use super::DecimalArrayExt;
use super::DecimalArraySlotsExt;
use super::DecimalPlugin;
use super::vtable::DecimalMetadata;
use crate::ArrayDeserialization;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::VTable;
use crate::VortexSessionExecute;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::is_sorted::is_sorted;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::Narrow;
use crate::arrays::NarrowArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::narrow::NarrowArraySlotsExt;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability;
use crate::dtype::i256;
use crate::dtype::integer::integer_dtype;
use crate::patches::Patches;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

#[rstest]
#[case::i8(2, DecimalType::I8)]
#[case::i16(3, DecimalType::I16)]
#[case::i32(5, DecimalType::I32)]
#[case::i64(10, DecimalType::I64)]
#[case::i128(19, DecimalType::I128)]
#[case::i256(39, DecimalType::I256)]
fn test_child_dtype_follows_precision(#[case] precision: u8, #[case] width: DecimalType) {
    let array = DecimalArray::from_option_iter(
        [Some(-1i8), None, Some(1)],
        DecimalDType::new(precision, 2),
    );
    assert_eq!(
        array.values_dtype(),
        &integer_dtype(width, Nullability::Nullable)
    );
    assert_eq!(array.as_ref().children().len(), 1);
    assert!(array.as_ref().buffer_handles().is_empty());
    assert_eq!(array.values_type(), DecimalType::I8);
    if width > DecimalType::I8 {
        assert!(array.values().is::<Narrow>());
    }
}

#[test]
fn test_rejects_wrong_logical_child_dtype() {
    assert!(
        DecimalArray::try_new_values(buffer![1i32, 2].into_array(), DecimalDType::new(39, 0),)
            .is_err()
    );
}

#[rstest]
#[case::i128_to_i64(DecimalArray::from_iter([i128::MAX], DecimalDType::new(10, 0)))]
#[case::i256_to_i128(DecimalArray::from_iter([i256::from_parts(0, 1)], DecimalDType::new(38, 0)))]
fn test_wider_native_input_cast_checks_overflow(#[case] array: DecimalArray) {
    let mut ctx = array_session().create_execution_ctx();
    let err = array.materialize_values(&mut ctx).unwrap_err();
    assert!(
        err.to_string().contains("Integer does not fit"),
        "got: {err}"
    );
}

#[test]
fn test_wider_native_input_materializes_at_logical_width() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let array = DecimalArray::from_iter([0i128, 1, -2], dtype).materialize_values(&mut ctx)?;
    assert_eq!(array.values_type(), DecimalType::I64);
    assert_arrays_eq!(
        array,
        DecimalArray::from_iter([0i64, 1, -2], dtype),
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case::primitive(DecimalArray::from_iter([-1i8, 1], DecimalDType::new(2, 0)))]
#[case::narrow_primitive(DecimalArray::from_option_iter([Some(-1i8), None], DecimalDType::new(10, 0)))]
#[case::wide(DecimalArray::from_iter([-1i128, 1], DecimalDType::new(38, 0)))]
#[case::narrow_wide(DecimalArray::from_option_iter([Some(-1i128), None], DecimalDType::new(76, 0)))]
fn test_materialization_reuses_native_children(#[case] array: DecimalArray) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let materialized = array.materialize_values(&mut ctx)?;
    assert!(ArrayRef::ptr_eq(array.as_ref(), materialized.as_ref()));

    Ok(())
}

#[test]
fn test_canonicalization_preserves_encoded_child() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(76, 2);
    let dictionary = DictArray::try_new(
        buffer![0u8, 1, 0, 1].into_array(),
        buffer![-123i32, 456].into_array(),
    )?
    .into_array();
    let values = NarrowArray::try_new(
        dictionary.clone(),
        integer_dtype(DecimalType::I256, Nullability::NonNullable),
    )?
    .into_array();
    let array = DecimalArray::try_new_values(values, dtype)?;
    let canonical = array
        .clone()
        .into_array()
        .execute::<DecimalArray>(&mut ctx)?;
    assert!(ArrayRef::ptr_eq(
        canonical.values().as_::<Narrow>().values(),
        &dictionary
    ));
    assert_eq!(canonical.values_type(), DecimalType::I32);
    assert!(canonical.as_ref().buffer_handles().is_empty());

    let selected = canonical
        .take(buffer![3u32, 0].into_array())?
        .execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(selected.values_type(), DecimalType::I32);
    assert_arrays_eq!(
        selected,
        DecimalArray::from_iter([456i32, -123], dtype),
        &mut ctx
    );

    let recursive = array
        .into_array()
        .execute::<RecursiveCanonical>(&mut ctx)?
        .0
        .into_decimal();
    assert!(recursive.values().is::<Narrow>());
    assert_eq!(recursive.buffer_handle().len(), 4 * size_of::<i32>());
    assert_eq!(recursive.buffer::<i32>(), buffer![-123i32, 456, -123, 456]);

    Ok(())
}

#[test]
fn test_wide_patch_widens_storage() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(39, 0);
    let value = 1i128 << 100;
    let patches = Patches::new(
        3,
        0,
        buffer![1u32].into_array(),
        DecimalArray::from_iter([value], dtype).into_array(),
        None,
    )?;
    let result = DecimalArray::from_iter([1i8, 2, 3], dtype).patch(&patches, &mut ctx)?;
    assert_eq!(
        result.values_dtype(),
        &integer_dtype(DecimalType::I256, Nullability::NonNullable)
    );
    assert_eq!(result.values_type(), DecimalType::I128);
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter([1, value, 3], dtype),
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case::nonnullable(Nullability::NonNullable)]
#[case::nullable(Nullability::Nullable)]
fn test_fill_null_accepts_a_wider_logical_value(
    #[case] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(39, 0);
    let value = 1i128 << 100;
    let array = DecimalArray::from_option_iter([Some(1i8), None, Some(-2)], dtype);
    let result = array
        .into_array()
        .fill_null(Scalar::decimal(
            DecimalValue::I128(value),
            dtype,
            nullability,
        ))?
        .execute::<DecimalArray>(&mut ctx)?
        .materialize_values(&mut ctx)?;
    assert_eq!(result.dtype(), &DType::Decimal(dtype, nullability));
    assert_eq!(result.values_type(), DecimalType::I128);
    assert_eq!(result.buffer_handle().len(), 3 * size_of::<i128>());
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter([1, value, -2], dtype)
            .into_array()
            .cast(DType::Decimal(dtype, nullability))?,
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case::same_precision(39)]
#[case::wider_precision(76)]
fn test_nonnullable_cast_checks_encoded_child_validity(
    #[case] precision: u8,
    #[values(true, false)] second_valid: bool,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let source = DecimalDType::new(39, 2);
    let target = DecimalDType::new(precision, 2);
    let dictionary = DictArray::try_new(
        buffer![0u8, 1].into_array(),
        PrimitiveArray::new(
            buffer![1i8, -2],
            Validity::Array(BoolArray::from_iter([true, second_valid]).into_array()),
        )
        .into_array(),
    )?;
    let values = NarrowArray::try_new(
        dictionary.into_array(),
        integer_dtype(DecimalType::I256, Nullability::Nullable),
    )?;
    let array = DecimalArray::try_new_values(values.into_array(), source)?;
    let result = array
        .into_array()
        .cast(DType::Decimal(target, Nullability::NonNullable))?
        .execute::<DecimalArray>(&mut ctx);

    if second_valid {
        assert_arrays_eq!(
            result?,
            DecimalArray::from_iter([1i8, -2], target),
            &mut ctx
        );
    } else {
        assert!(result.is_err());
    }

    Ok(())
}

#[rstest]
#[case::same_integer_width(18, DecimalType::I64)]
#[case::wider_integer_width(39, DecimalType::I256)]
fn test_precision_widening_cast_preserves_encoded_child(
    #[case] precision: u8,
    #[case] width: DecimalType,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dictionary = DictArray::try_new(
        buffer![0u8, 1, 0].into_array(),
        buffer![1i32, -2].into_array(),
    )?
    .into_array();
    let values = NarrowArray::try_new(
        dictionary.clone(),
        integer_dtype(DecimalType::I64, Nullability::NonNullable),
    )?;
    let source = DecimalArray::try_new_values(values.into_array(), DecimalDType::new(10, 2))?;
    let target = DecimalDType::new(precision, 2);
    let casted = source
        .into_array()
        .cast(DType::Decimal(target, Nullability::NonNullable))?;
    let decimal = casted
        .as_opt::<Decimal>()
        .vortex_expect("Widening precision reduces without executing the child");
    assert_eq!(
        decimal.values_dtype(),
        &integer_dtype(width, Nullability::NonNullable)
    );
    assert!(ArrayRef::ptr_eq(
        decimal.values().as_::<Narrow>().values(),
        &dictionary
    ));
    assert_arrays_eq!(
        casted,
        DecimalArray::from_iter([1i32, -2, 1], target),
        &mut ctx
    );

    Ok(())
}

#[test]
fn test_aggregates_and_comparisons_keep_decimal_semantics() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(39, 2);
    let array = DecimalArray::from_iter([-128i8, -1, 0, 127], dtype).into_array();
    let bounds = min_max(&array, &mut ctx, NumericalAggregateOpts::default())?
        .vortex_expect("Non-empty bounds");
    assert_eq!(
        bounds.min,
        Scalar::decimal(DecimalValue::I8(-128), dtype, Nullability::NonNullable)
    );
    assert_eq!(
        bounds.max,
        Scalar::decimal(DecimalValue::I8(127), dtype, Nullability::NonNullable)
    );
    assert!(is_sorted(&array, &mut ctx)?);
    let rhs = ConstantArray::new(
        Scalar::decimal(DecimalValue::I128(1000), dtype, Nullability::NonNullable),
        4,
    )
    .into_array();
    assert_arrays_eq!(
        array.binary(rhs, Operator::Lt)?,
        BoolArray::from_iter([true; 4]),
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case::above_storage(1000i128, 2000i128)]
#[case::below_storage(-2000i128, -1000i128)]
fn test_between_outside_storage_range_preserves_nulls(
    #[case] lower: i128,
    #[case] upper: i128,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(39, 0);
    let array = DecimalArray::from_option_iter([Some(-1i8), None, Some(1)], dtype);
    let bound = |value| {
        ConstantArray::new(
            Scalar::decimal(DecimalValue::I128(value), dtype, Nullability::NonNullable),
            array.len(),
        )
        .into_array()
    };
    let lower = bound(lower);
    let upper = bound(upper);
    assert_arrays_eq!(
        array.into_array().between(
            lower,
            upper,
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?,
        BoolArray::from_iter([Some(false), None, Some(false)]),
        &mut ctx
    );

    Ok(())
}

#[test]
fn test_partial_aggregates_preserve_decimal_boundaries() -> VortexResult<()> {
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    let dtype = DecimalDType::new(76, 2);
    let input_dtype = DType::Decimal(dtype, Nullability::Nullable);
    for aggregate in [
        MinMax.bind(NumericalAggregateOpts::default()),
        IsSorted.bind(IsSortedOptions { strict: false }),
        IsConstant.bind(EmptyOptions),
    ] {
        assert!(
            session
                .aggregate_fns()
                .find_aggregate_kernel(VTable::id(&Decimal), aggregate.id())
                .is_some()
        );

        let mut expected = aggregate.accumulator(&input_dtype)?;
        let mut actual = aggregate.accumulator(&input_dtype)?;
        for values in [vec![None, Some(-128i8), Some(0)], vec![Some(0), Some(127)]] {
            let compact = DecimalArray::from_option_iter(values.clone(), dtype);
            let wide = DecimalArray::from_option_iter(
                values
                    .into_iter()
                    .map(|value| value.map(|value| i256::from_i128(i128::from(value)))),
                dtype,
            );
            actual.accumulate(&compact.into_array(), &mut ctx)?;
            expected.accumulate(&wide.into_array(), &mut ctx)?;
            assert_eq!(actual.partial_scalar()?, expected.partial_scalar()?);
        }
        assert_eq!(actual.final_scalar()?, expected.final_scalar()?);
    }

    Ok(())
}

#[test]
fn test_sum_uses_the_decimal_kernel() -> VortexResult<()> {
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    let dtype = DecimalDType::new(39, 2);
    let array = DecimalArray::from_iter([10i8, 20], dtype).into_array();
    let aggregate = Sum.bind(NumericalAggregateOpts::default());
    let kernel = session
        .aggregate_fns()
        .find_aggregate_kernel(VTable::id(&Decimal), aggregate.id())
        .vortex_expect("Decimal registers a fallback aggregate kernel");
    assert!(kernel.aggregate(&aggregate, &array, &mut ctx)?.is_none());

    let mut accumulator = aggregate.accumulator(array.dtype())?;
    accumulator.accumulate(&array, &mut ctx)?;
    assert_eq!(
        accumulator.final_scalar()?,
        Scalar::decimal(
            DecimalValue::I8(30),
            DecimalDType::new(49, 2),
            Nullability::Nullable,
        )
    );

    Ok(())
}

#[test]
fn test_legacy_wire_keeps_narrow_storage() -> VortexResult<()> {
    let session = &array_session();
    let array =
        DecimalArray::from_option_iter([Some(10i32), None, Some(-20)], DecimalDType::new(76, 2))
            .into_array();
    let wire = DecimalPlugin
        .serialize(&array, session)?
        .vortex_expect("Serializable decimal");
    assert_eq!(wire.buffers[0].len(), 3 * size_of::<i32>());
    assert_eq!(
        DecimalMetadata::decode(wire.metadata.as_slice())?.values_type,
        DecimalType::I32 as i32
    );
    let buffers = wire
        .buffers
        .into_iter()
        .map(BufferHandle::new_host)
        .collect::<Vec<_>>();
    let decoded = DecimalPlugin.deserialize(
        ArrayDeserialization::new(
            VTable::id(&Decimal),
            array.dtype(),
            array.len(),
            &wire.metadata,
            &buffers,
            &wire.children,
        ),
        session,
    )?;
    assert_eq!(decoded.as_::<Decimal>().values_type(), DecimalType::I32);
    assert_arrays_eq!(decoded, array, &mut session.create_execution_ctx());

    Ok(())
}

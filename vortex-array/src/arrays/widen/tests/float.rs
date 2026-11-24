// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Construction, execution, compute, and aggregate tests for floating-point [`Widen`] arrays.

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::SESSION;
use super::assert_compare_matches_canonical;
use super::canonical;
use super::stored_ptype;
use super::widen;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyArrayData;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum_v2::SumV2;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::WidenSlots;
use crate::arrays::widen::compute::aggregate::WidenArrayAggregate;
use crate::arrays::widen::compute::constant_fits_values;
use crate::arrays::widen::compute::ptype_holding_both_values;
use crate::arrays::widen::compute::ptype_holding_values_and_scalar;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::half::f16;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;
use crate::scalar_fn::builtin::operators::Operator;
use crate::validity::Validity;

const ALL_COMPARE_OPERATORS: [Operator; 6] = [
    Operator::Eq,
    Operator::NotEq,
    Operator::Lt,
    Operator::Lte,
    Operator::Gt,
    Operator::Gte,
];

/// Edge-case `f32` values covering negative NaN, infinities, finite bounds, subnormals, signed
/// zeros, and positive NaN, sorted in IEEE-754 total order.
fn f32_edge_values() -> Vec<f32> {
    vec![
        -f32::NAN,
        f32::NEG_INFINITY,
        -f32::MAX,
        -100.0,
        -0.1_f32,
        -f32::from_bits(1),
        -0.0_f32,
        0.0_f32,
        f32::from_bits(1),
        0.1_f32,
        1.0,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ]
}

/// Edge-case `f16` values sorted in IEEE-754 total order.
fn f16_edge_values() -> Vec<f16> {
    vec![
        -f16::NAN,
        f16::NEG_INFINITY,
        f16::MIN,
        f16::from_f32(-1.5),
        -f16::from_bits(1),
        f16::NEG_ZERO,
        f16::ZERO,
        f16::from_bits(1),
        f16::from_f32(1.5),
        f16::MAX,
        f16::INFINITY,
        f16::NAN,
    ]
}

#[rstest]
#[case::f16_to_f32(PType::F16, PType::F32, Nullability::NonNullable)]
#[case::f16_to_f64(PType::F16, PType::F64, Nullability::Nullable)]
#[case::f32_to_f64(PType::F32, PType::F64, Nullability::NonNullable)]
#[case::f32_to_f64_nullable(PType::F32, PType::F64, Nullability::Nullable)]
fn test_try_new_accepts_narrower_float_dtypes(
    #[case] child: PType,
    #[case] logical: PType,
    #[case] nullability: Nullability,
) -> VortexResult<()> {
    let values = Canonical::empty(&DType::Primitive(child, nullability)).into_array();
    let widened = WidenArray::try_new(values, DType::Primitive(logical, nullability))?;

    assert_eq!(widened.dtype(), &DType::Primitive(logical, nullability));
    assert_eq!(
        widened.values().dtype(),
        &DType::Primitive(child, nullability)
    );

    Ok(())
}

#[rstest]
#[case::equal_f16(PType::F16.into(), PType::F16.into(), "narrower")]
#[case::equal_f32(PType::F32.into(), PType::F32.into(), "narrower")]
#[case::equal_f64(PType::F64.into(), PType::F64.into(), "narrower")]
#[case::wider_f64_to_f32(PType::F64.into(), PType::F32.into(), "narrower")]
#[case::wider_f32_to_f16(PType::F32.into(), PType::F16.into(), "narrower")]
#[case::int_to_float(PType::I16.into(), PType::F64.into(), "integer, float, or decimal dtypes")]
#[case::float_to_int(PType::F32.into(), PType::I64.into(), "integer, float, or decimal dtypes")]
#[case::float_to_decimal(
    PType::F32.into(),
    DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable),
    "integer, float, or decimal dtypes",
)]
#[case::nullability_mismatch(
    PType::F32.into(),
    DType::Primitive(PType::F64, Nullability::Nullable),
    "nullability",
)]
fn test_rejects_invalid_float_dtypes(
    #[case] values: DType,
    #[case] logical: DType,
    #[case] requirement: &str,
) {
    let values = Canonical::empty(&values).into_array();
    let slots = WidenSlots {
        values: values.clone(),
    }
    .into_slots();
    let parts = ArrayParts::new(Widen, logical.clone(), 0, EmptyArrayData, slots);

    let try_new_error = WidenArray::try_new(values, logical)
        .expect_err("invalid float Widen dtypes must fail")
        .to_string();
    let validate_error = WidenArray::try_from_parts(parts)
        .expect_err("invalid float Widen dtypes must fail validation")
        .to_string();

    assert!(try_new_error.contains(requirement), "{try_new_error}");
    assert!(validate_error.contains(requirement), "{validate_error}");
}

#[test]
fn test_try_new_and_reduce_flatten_nested_float_widen() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_option_iter([
        Some(f16::NEG_ZERO),
        None,
        Some(f16::from_f32(1.5)),
        Some(f16::INFINITY),
    ])
    .into_array();

    let inner = widen(values.clone(), PType::F32).into_array();
    let outer = WidenArray::try_new(
        inner.clone(),
        DType::Primitive(PType::F64, Nullability::Nullable),
    )?;
    assert!(outer.values().is::<Primitive>());
    assert_eq!(
        outer.values().dtype(),
        &DType::Primitive(PType::F16, Nullability::Nullable)
    );

    let nested = WidenArray::try_from_parts(ArrayParts::new(
        Widen,
        DType::Primitive(PType::F64, Nullability::Nullable),
        inner.len(),
        EmptyArrayData,
        WidenSlots { values: inner }.into_slots(),
    ))?;
    assert!(nested.values().is::<Widen>());
    let flattened = nested.into_array().optimize_ctx(&SESSION)?;
    assert!(ArrayRef::ptr_eq(flattened.as_::<Widen>().values(), &values));

    let expected = PrimitiveArray::from_option_iter([
        Some(-0.0_f64),
        None,
        Some(1.5_f64),
        Some(f64::INFINITY),
    ]);
    assert_arrays_eq!(outer, expected, &mut ctx);
    assert_arrays_eq!(flattened, expected, &mut ctx);

    Ok(())
}

#[test]
fn test_encode_float_f64_bit_exact_round_trip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    // All values representable in f16 -> narrows to f16.
    let f16_in_f64: Vec<f64> = f16_edge_values().into_iter().map(f16::to_f64).collect();
    let input = PrimitiveArray::from_iter(f16_in_f64.clone());
    let encoded = WidenArray::encode_float(input.clone(), &mut ctx)?;
    assert_eq!(stored_ptype(&encoded), PType::F16);
    assert_eq!(encoded.dtype(), input.dtype());
    let decoded = encoded.clone().execute::<PrimitiveArray>(&mut ctx)?;
    for (idx, (&actual, &expected)) in decoded
        .as_slice::<f64>()
        .iter()
        .zip(&f16_in_f64)
        .enumerate()
    {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "f16->f64 bit mismatch at row {idx}"
        );
        let scalar = encoded.execute_scalar(idx, &mut ctx)?;
        assert_eq!(
            scalar.as_primitive().typed_value::<f64>().map(f64::to_bits),
            Some(expected.to_bits()),
            "f16->f64 scalar_at bit mismatch at row {idx}"
        );
    }

    // Values representable in f32 (including -0.0, subnormals, canonical NaN) -> narrows to f32.
    let f32_in_f64: Vec<f64> = f32_edge_values().into_iter().map(f64::from).collect();
    let input = PrimitiveArray::from_iter(f32_in_f64.clone());
    let encoded = WidenArray::encode_float(input, &mut ctx)?;
    assert_eq!(stored_ptype(&encoded), PType::F32);
    let decoded = encoded.execute::<PrimitiveArray>(&mut ctx)?;
    for (idx, (&actual, &expected)) in decoded
        .as_slice::<f64>()
        .iter()
        .zip(&f32_in_f64)
        .enumerate()
    {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "f32->f64 bit mismatch at row {idx}"
        );
    }

    Ok(())
}

#[rstest]
#[case::inexact_decimal(vec![0.0_f64, 0.1_f64, 1.0_f64])]
#[case::overflow_f32(vec![1.0_f64, 1e100_f64])]
#[case::underflow_f32(vec![1.0_f64, 1e-100_f64])]
#[case::f64_max(vec![f64::MAX])]
#[case::non_f32_qnan_payload(vec![1.0_f64, f64::from_bits(0x7ff8_0000_0000_0001)])]
#[case::f64_snan_low(vec![1.0_f64, f64::from_bits(0x7ff0_0000_0000_0001)])]
#[case::f64_snan_aligned(vec![1.0_f64, f64::from_bits(0x7ff0_0000_2000_0000)])]
fn test_encode_float_f64_rejects_lossy_and_snan(#[case] values: Vec<f64>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = PrimitiveArray::from_iter(values);
    let encoded = WidenArray::encode_float(input.clone(), &mut ctx)?;

    assert_eq!(stored_ptype(&encoded), PType::F64);
    assert!(!encoded.is::<Widen>());
    assert_arrays_eq!(encoded, input, &mut ctx);

    Ok(())
}

#[rstest]
#[case::f16_exact(
    PrimitiveArray::from_iter(f16_edge_values().into_iter().map(f16::to_f32).collect::<Vec<_>>()),
    PType::F16,
)]
#[case::f32_inexact(PrimitiveArray::from_iter([0.1_f32, 1.0_f32]), PType::F32)]
#[case::f32_non_f16_qnan(PrimitiveArray::from_iter([1.0_f32, f32::from_bits(0x7fc0_0001)]), PType::F32)]
#[case::f32_snan(PrimitiveArray::from_iter([1.0_f32, f32::from_bits(0x7f80_2000)]), PType::F32)]
#[case::f16_unchanged(PrimitiveArray::from_iter([f16::ONE, f16::NEG_ZERO]), PType::F16)]
#[case::empty_f64(PrimitiveArray::empty::<f64>(Nullability::NonNullable), PType::F16)]
#[case::empty_f32(PrimitiveArray::empty::<f32>(Nullability::Nullable), PType::F16)]
#[case::all_null_f64(
    PrimitiveArray::new(buffer![0.1_f64, f64::from_bits(0x7ff0_0000_0000_0001)], Validity::AllInvalid),
    PType::F16,
)]
#[case::null_payloads_ignored_and_zeroed(
    PrimitiveArray::new(
        buffer![0.1_f64, 1.5_f64, f64::from_bits(0x7ff0_0000_0000_0001)],
        Validity::from_iter([false, true, false]),
    ),
    PType::F16,
)]
fn test_encode_float_selection(
    #[case] input: PrimitiveArray,
    #[case] expected_stored: PType,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = WidenArray::encode_float(input.clone(), &mut ctx)?;

    assert_eq!(stored_ptype(&encoded), expected_stored);
    assert_eq!(encoded.dtype(), input.dtype());
    assert_arrays_eq!(encoded, input, &mut ctx);

    // Null slots in a narrowed child are zeroed so signaling NaNs cannot leak into the child.
    if let Some(w) = encoded.as_opt::<Widen>()
        && w.values().dtype().as_ptype() == PType::F16
        && input.len() == 3
    {
        let child = w.values().as_::<Primitive>();
        assert_eq!(child.as_slice::<f16>()[0].to_bits(), 0);
        assert_eq!(child.as_slice::<f16>()[2].to_bits(), 0);
    }

    Ok(())
}

#[test]
fn test_encode_float_rejects_integer() {
    let mut ctx = SESSION.create_execution_ctx();
    let error = WidenArray::encode_float(PrimitiveArray::from_iter([1i32, 2, 3]), &mut ctx)
        .expect_err("encode_float must reject integers");
    assert!(error.to_string().contains("floating-point"), "{error}");
}

#[test]
fn test_float_execute_encoded_dict_child() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let codes = buffer![2u8, 0, 1, 2].into_array();
    let values = DictArray::try_new(
        codes.clone(),
        buffer![-0.0_f32, 0.0_f32, 1.5_f32].into_array(),
    )?;
    let array = WidenArray::try_new(values.into_array(), PType::F64.into())?;

    let step = array
        .clone()
        .into_array()
        .execute::<ArrayRef>(&mut ctx)?
        .optimize_ctx(&SESSION)?;
    assert!(step.is::<Dict>());
    assert!(ArrayRef::ptr_eq(step.as_::<Dict>().codes(), &codes));
    assert_eq!(step.dtype(), &DType::from(PType::F64));
    assert_arrays_eq!(
        array,
        PrimitiveArray::from_iter([1.5_f64, -0.0, 0.0, 1.5]),
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case::non_nullable(widen(Buffer::from(f32_edge_values()), PType::F64).into_array())]
#[case::nullable(
    widen(
        PrimitiveArray::from_option_iter(
            f32_edge_values()
                .into_iter()
                .enumerate()
                .map(|(i, v)| (i % 4 != 1).then_some(v))
        ),
        PType::F64,
    )
    .into_array()
)]
#[case::all_invalid(
    widen(
        PrimitiveArray::new(Buffer::from(f32_edge_values()), Validity::AllInvalid),
        PType::F64,
    )
    .into_array()
)]
fn test_compare_f32_widen_to_f64_constants(#[case] array: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let constants = [
        // Exact f32 values
        f64::NEG_INFINITY,
        f64::from(-f32::MAX),
        -0.0_f64,
        0.0_f64,
        f64::from(f32::from_bits(1)),
        f64::from(0.1_f32),
        1.0_f64,
        f64::from(f32::MAX),
        f64::INFINITY,
        // Strictly between adjacent f32 values, including beyond ±f32::MAX and inside ±min_sub
        -1e100_f64,
        f64::from(-f32::MAX).next_down(),
        f64::from(-f32::MAX).next_up(),
        -0.1_f64,
        -1e-45_f64,
        -1e-100_f64,
        1e-100_f64,
        1e-45_f64,
        0.1_f64,
        f64::from(f32::MAX).next_down(),
        f64::from(f32::MAX).next_up(),
        1e100_f64,
        // NaN variants (fallback to canonical)
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7ff8_0000_0000_0001),
        f64::from_bits(0x7ff0_0000_0000_0001),
        f64::from_bits(0xfff8_0000_0000_0001),
    ];

    for c in constants {
        for nullability in [Nullability::NonNullable, Nullability::Nullable] {
            let rhs =
                ConstantArray::new(Scalar::primitive(c, nullability), array.len()).into_array();
            for op in ALL_COMPARE_OPERATORS {
                assert_compare_matches_canonical(&array, &rhs, op, &mut ctx)?;
            }
        }
    }

    let null_rhs = ConstantArray::new(
        Scalar::null(DType::Primitive(PType::F64, Nullability::Nullable)),
        array.len(),
    )
    .into_array();
    for op in ALL_COMPARE_OPERATORS {
        assert_compare_matches_canonical(&array, &null_rhs, op, &mut ctx)?;
    }

    Ok(())
}

#[test]
fn test_compare_f16_widen_constants_and_constant_folding() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let f16_to_f64 = widen(Buffer::from(f16_edge_values()), PType::F64).into_array();
    let f64_constants = [
        f64::NEG_INFINITY,
        -1e10_f64,
        f16::MIN.to_f64().next_down(),
        f16::MIN.to_f64(),
        f16::MIN.to_f64().next_up(),
        -0.1_f64,
        -1e-10_f64,
        -0.0_f64,
        0.0_f64,
        1e-10_f64,
        0.1_f64,
        f16::MAX.to_f64().next_down(),
        f16::MAX.to_f64(),
        f16::MAX.to_f64().next_up(),
        1e10_f64,
        f64::INFINITY,
        f64::NAN,
        -f64::NAN,
    ];
    for c in f64_constants {
        let rhs = ConstantArray::new(c, f16_to_f64.len()).into_array();
        for op in ALL_COMPARE_OPERATORS {
            assert_compare_matches_canonical(&f16_to_f64, &rhs, op, &mut ctx)?;
        }
    }

    let f16_to_f32 = widen(Buffer::from(f16_edge_values()), PType::F32).into_array();
    let f32_constants = [
        f32::NEG_INFINITY,
        -1e10_f32,
        f16::MIN.to_f32(),
        -0.1_f32,
        -1e-10_f32,
        -0.0_f32,
        0.0_f32,
        1e-10_f32,
        0.1_f32,
        f16::MAX.to_f32(),
        1e10_f32,
        f32::INFINITY,
        f32::NAN,
        -f32::NAN,
    ];
    for c in f32_constants {
        let rhs = ConstantArray::new(c, f16_to_f32.len()).into_array();
        for op in ALL_COMPARE_OPERATORS {
            assert_compare_matches_canonical(&f16_to_f32, &rhs, op, &mut ctx)?;
        }
    }

    // Non-nullable Eq/NotEq against an in-between float constant folds to a ConstantArray.
    let in_between = ConstantArray::new(0.1_f64, f16_to_f64.len()).into_array();
    let eq_result = f16_to_f64
        .binary(in_between, Operator::Eq)?
        .execute::<ArrayRef>(&mut ctx)?;
    assert!(eq_result.is::<Constant>());

    Ok(())
}

#[test]
fn test_compare_two_float_widens() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = widen(
        PrimitiveArray::from_option_iter([
            Some(-f16::NAN),
            Some(f16::NEG_INFINITY),
            Some(f16::NEG_ZERO),
            Some(f16::ZERO),
            Some(f16::ONE),
            None,
            Some(f16::INFINITY),
            Some(f16::NAN),
        ]),
        PType::F64,
    )
    .into_array();
    let rhs = widen(
        PrimitiveArray::from_option_iter([
            Some(f32::NAN),
            Some(f32::NEG_INFINITY),
            Some(0.0_f32),
            Some(-0.0_f32),
            Some(1.0_f32),
            Some(2.0_f32),
            None,
            Some(f32::NAN),
        ]),
        PType::F64,
    )
    .into_array();

    assert_eq!(
        ptype_holding_both_values(lhs.as_::<Widen>(), rhs.as_::<Widen>()),
        Some(PType::F32)
    );
    for op in ALL_COMPARE_OPERATORS {
        assert_compare_matches_canonical(&lhs, &rhs, op, &mut ctx)?;
    }

    Ok(())
}

#[test]
fn test_float_cast_rules() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let f16_to_f64 = widen(
        PrimitiveArray::from_option_iter([
            Some(f16::from_f32(-2.0)),
            None,
            Some(f16::from_f32(3.0)),
        ]),
        PType::F64,
    )
    .into_array();

    // Strictly wider float keeps Widen.
    let to_f32 = f16_to_f64
        .cast(DType::Primitive(PType::F32, Nullability::Nullable))?
        .optimize_ctx(&SESSION)?;
    assert_eq!(stored_ptype(&to_f32), PType::F16);
    assert_arrays_eq!(
        to_f32,
        canonical(&f16_to_f64, &mut ctx)?
            .cast(DType::Primitive(PType::F32, Nullability::Nullable))?,
        &mut ctx
    );

    // Same width as child unwraps Widen.
    let to_f16 = f16_to_f64
        .cast(DType::Primitive(PType::F16, Nullability::Nullable))?
        .optimize_ctx(&SESSION)?;
    assert!(!to_f16.is::<Widen>());
    assert_arrays_eq!(
        to_f16,
        canonical(&f16_to_f64, &mut ctx)?
            .cast(DType::Primitive(PType::F16, Nullability::Nullable))?,
        &mut ctx
    );

    // Float-to-int cast delegates to child and matches canonical.
    let to_i32 = f16_to_f64
        .cast(DType::Primitive(PType::I32, Nullability::Nullable))?
        .optimize_ctx(&SESSION)?;
    assert_arrays_eq!(
        to_i32,
        canonical(&f16_to_f64, &mut ctx)?
            .cast(DType::Primitive(PType::I32, Nullability::Nullable))?,
        &mut ctx
    );

    // Float Widen (F32 -> F64) narrowed to F16 matches canonical f64 -> f16 rounding.
    let f32_to_f64 = widen(
        buffer![0.1_f32, -0.0_f32, 100_000.0_f32, f32::NAN],
        PType::F64,
    )
    .into_array();
    let to_f16 = f32_to_f64.cast(PType::F16.into())?;
    let expected = canonical(&f32_to_f64, &mut ctx)?.cast(PType::F16.into())?;
    assert_arrays_eq!(to_f16, expected, &mut ctx);

    // Out-of-range float-to-int fails just like canonical.
    let big = widen(buffer![1e20_f32], PType::F64).into_array();
    assert!(
        big.cast(PType::I32.into())?
            .execute::<Canonical>(&mut ctx)
            .is_err()
    );

    // Float-to-decimal fails just like canonical.
    let dec = DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable);
    assert!(
        f32_to_f64
            .cast(dec)?
            .execute::<Canonical>(&mut ctx)
            .is_err()
    );

    Ok(())
}

#[rstest]
#[case::fits_f16(Scalar::primitive(0.5_f64, Nullability::NonNullable), PType::F16)]
#[case::neg_zero(Scalar::primitive(-0.0_f64, Nullability::NonNullable), PType::F16)]
#[case::infinity(Scalar::primitive(f64::INFINITY, Nullability::NonNullable), PType::F16)]
#[case::quiet_nan(Scalar::primitive(f64::NAN, Nullability::NonNullable), PType::F16)]
#[case::needs_f32(Scalar::primitive(100_000.0_f64, Nullability::NonNullable), PType::F32)]
#[case::needs_f64(Scalar::primitive(0.1_f64, Nullability::NonNullable), PType::F64)]
#[case::snan_needs_f64(
    Scalar::primitive(f64::from_bits(0x7ff0_0000_0000_0001), Nullability::NonNullable),
    PType::F64
)]
fn test_float_fill_null(
    #[case] fill_value: Scalar,
    #[case] expected_stored: PType,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = widen(
        PrimitiveArray::from_option_iter([Some(f16::from_f32(-1.5)), None, Some(f16::ONE)]),
        PType::F64,
    )
    .into_array();

    assert_eq!(
        ptype_holding_values_and_scalar(array.as_::<Widen>(), &fill_value),
        Some(expected_stored)
    );

    let filled = array
        .clone()
        .fill_null(fill_value.clone())?
        .optimize_ctx(&SESSION)?;
    assert_eq!(stored_ptype(&filled), expected_stored);
    let expected = canonical(&array, &mut ctx)?.fill_null(fill_value)?;
    assert_arrays_eq!(filled, expected, &mut ctx);

    Ok(())
}

#[test]
fn test_ptype_holding_values_and_null_float_scalar() {
    let array = widen(
        PrimitiveArray::from_option_iter([Some(f16::from_f32(-1.5)), None, Some(f16::ONE)]),
        PType::F64,
    );
    let null_scalar = Scalar::null(DType::Primitive(PType::F64, Nullability::Nullable));
    assert_eq!(
        ptype_holding_values_and_scalar(array.as_view(), &null_scalar),
        Some(PType::F16)
    );
}

#[test]
fn test_float_filter_mask_slice_take() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = widen(
        PrimitiveArray::from_option_iter([
            Some(-0.0_f32),
            None,
            Some(1.5_f32),
            Some(f32::INFINITY),
            Some(f32::NAN),
        ]),
        PType::F64,
    )
    .into_array();
    let expected_canonical = canonical(&array, &mut ctx)?;

    let filter_mask = Mask::from_iter([true, false, true, true, true]);
    let filtered = array.filter(filter_mask.clone())?.optimize_ctx(&SESSION)?;
    assert_eq!(stored_ptype(&filtered), PType::F32);
    assert_arrays_eq!(filtered, expected_canonical.filter(filter_mask)?, &mut ctx);

    let keep_mask = BoolArray::from_iter([true, true, false, true, true]).into_array();
    let masked = array
        .clone()
        .mask(keep_mask.clone())?
        .optimize_ctx(&SESSION)?;
    assert_eq!(stored_ptype(&masked), PType::F32);
    assert_arrays_eq!(
        masked,
        expected_canonical.clone().mask(keep_mask)?,
        &mut ctx
    );

    let sliced = array.slice(1..4)?.optimize_ctx(&SESSION)?;
    assert_eq!(stored_ptype(&sliced), PType::F32);
    assert_arrays_eq!(sliced, expected_canonical.slice(1..4)?, &mut ctx);

    let indices = buffer![4u32, 0, 1, 3].into_array();
    let taken = array.take(indices.clone())?.execute::<ArrayRef>(&mut ctx)?;
    assert_eq!(stored_ptype(&taken), PType::F32);
    assert_arrays_eq!(taken, expected_canonical.take(indices)?, &mut ctx);

    Ok(())
}

#[test]
fn test_float_constant_fits_values() {
    let f32_dtype = DType::from(PType::F32);
    assert_eq!(
        constant_fits_values(
            &Scalar::primitive(1.5_f64, Nullability::NonNullable),
            &f32_dtype
        ),
        Some(true)
    );
    assert_eq!(
        constant_fits_values(
            &Scalar::primitive(-0.0_f64, Nullability::NonNullable),
            &f32_dtype
        ),
        Some(true)
    );
    assert_eq!(
        constant_fits_values(
            &Scalar::primitive(f64::INFINITY, Nullability::NonNullable),
            &f32_dtype
        ),
        Some(true)
    );
    assert_eq!(
        constant_fits_values(
            &Scalar::primitive(0.1_f64, Nullability::NonNullable),
            &f32_dtype
        ),
        Some(false)
    );
    assert_eq!(
        constant_fits_values(
            &Scalar::primitive(f64::NAN, Nullability::NonNullable),
            &f32_dtype
        ),
        None
    );
    assert_eq!(
        constant_fits_values(
            &Scalar::null(DType::Primitive(PType::F64, Nullability::Nullable)),
            &f32_dtype,
        ),
        None
    );
}

#[test]
fn test_float_numeric_declines_and_preserves_f64_precision() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // 16_777_216.0 + 1.0 loses precision in f32 (rounds to 16_777_216.0), and 1e20 * 1e20 overflows
    // f32 to +inf, while both are exact/finite in f64.
    let lhs = widen(buffer![16_777_216.0_f32, 1e20_f32], PType::F64).into_array();
    let rhs = widen(buffer![1.0_f32, 1e20_f32], PType::F64).into_array();

    for op in [Operator::Add, Operator::Sub, Operator::Mul, Operator::Div] {
        let actual = lhs.binary(rhs.clone(), op)?;
        let expected = canonical(&lhs, &mut ctx)?.binary(canonical(&rhs, &mut ctx)?, op)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
    }

    // Double-check the exact f64 result values didn't suffer f32 rounding/overflow.
    let sum = lhs
        .binary(rhs.clone(), Operator::Add)?
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(sum.as_slice::<f64>()[0], 16_777_217.0_f64);
    let prod = lhs
        .binary(rhs, Operator::Mul)?
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert!(prod.as_slice::<f64>()[1].is_finite());

    Ok(())
}

#[test]
fn test_float_aggregates_match_canonical_and_sum_declines() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let skip_nans = NumericalAggregateOpts::default();
    let include_nans = NumericalAggregateOpts { skip_nans: false };

    let aggregates = [
        (MinMax.bind(skip_nans), true),
        (MinMax.bind(include_nans), true),
        (Min.bind(skip_nans), true),
        (Min.bind(include_nans), true),
        (Max.bind(skip_nans), true),
        (Max.bind(include_nans), true),
        (IsConstant.bind(EmptyOptions), true),
        (IsSorted.bind(IsSortedOptions { strict: false }), true),
        (IsSorted.bind(IsSortedOptions { strict: true }), true),
        (Sum.bind(skip_nans), false),
        (Sum.bind(include_nans), false),
        (SumV2.bind(skip_nans), false),
    ];

    let fixtures = [
        widen(Buffer::from(f32_edge_values()), PType::F64).into_array(),
        widen(buffer![-0.0_f32, 0.0_f32], PType::F64).into_array(),
        widen(buffer![-0.0_f32, -0.0_f32], PType::F64).into_array(),
        widen(buffer![0.0_f32, -0.0_f32], PType::F64).into_array(),
        widen(buffer![f32::NAN, f32::NAN], PType::F64).into_array(),
        widen(
            buffer![
                16_777_216.0_f32,
                1.0_f32,
                1.0_f32,
                1e38_f32,
                1e38_f32,
                1e38_f32,
                1e38_f32
            ],
            PType::F64,
        )
        .into_array(),
        widen(
            PrimitiveArray::from_option_iter([
                None,
                Some(f16::NEG_ZERO),
                Some(f16::ZERO),
                Some(f16::ONE),
            ]),
            PType::F64,
        )
        .into_array(),
        widen(
            PrimitiveArray::new(buffer![1.0_f32, 2.0_f32], Validity::AllInvalid),
            PType::F64,
        )
        .into_array(),
        widen(
            PrimitiveArray::empty::<f16>(Nullability::NonNullable),
            PType::F32,
        )
        .into_array(),
    ];

    for fixture in fixtures {
        let canonical_fixture = canonical(&fixture, &mut ctx)?;
        for (aggregate_fn, expected_served) in &aggregates {
            let kernel_result = WidenArrayAggregate.aggregate(aggregate_fn, &fixture, &mut ctx)?;
            assert_eq!(
                kernel_result.is_some(),
                *expected_served,
                "unexpected kernel coverage for {aggregate_fn} on {}",
                fixture.dtype()
            );

            let mut actual_acc = aggregate_fn.accumulator(fixture.dtype())?;
            actual_acc.accumulate(&fixture, &mut ctx)?;
            let actual = actual_acc.finish()?;

            let mut expected_acc = aggregate_fn.accumulator(canonical_fixture.dtype())?;
            expected_acc.accumulate(&canonical_fixture, &mut ctx)?;
            let expected = expected_acc.finish()?;

            assert_eq!(actual, expected, "mismatch for {aggregate_fn}");
        }
    }

    // Multi-chunk IsSorted merges partial structs across float Widen chunks cleanly.
    let chunk1 = widen(buffer![-0.0_f32, 0.0_f32], PType::F64).into_array();
    let chunk2 = widen(buffer![1.0_f32, 2.0_f32], PType::F64).into_array();
    let chunked = ChunkedArray::try_new(vec![chunk1, chunk2], PType::F64.into())?.into_array();
    let is_strict = IsSorted.bind(IsSortedOptions { strict: true });
    let mut acc = is_strict.accumulator(chunked.dtype())?;
    acc.accumulate(&chunked, &mut ctx)?;
    assert_eq!(acc.finish()?, Scalar::from(Some(true)));

    Ok(())
}

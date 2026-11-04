// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parity between [`Scalar`] equality/ordering and the array compare kernels.
//!
//! For every pair of representative values of a dtype, the result of comparing the two values as
//! rows of canonical arrays must agree with [`Scalar`]'s [`PartialEq`] and [`PartialOrd`]:
//!
//! - a row is null exactly when either operand is a top-level null;
//! - otherwise `op(lhs, rhs)` equals the operator applied to `lhs.partial_cmp(&rhs)`;
//! - `Eq` and `NotEq` also agree with `lhs == rhs` and `lhs != rhs`.
//!
//! Both array-array and array-constant comparisons are checked, since constants take separate
//! kernel paths from fully materialized operands.

use std::cmp::Ordering;
use std::sync::Arc;

use rstest::rstest;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::SESSION;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::builders::builder_with_capacity_in;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::Nullability::Nullable;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::dtype::half::f16;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::memory::MemorySessionExt;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

const OPERATORS: [CompareOperator; 6] = [
    CompareOperator::Eq,
    CompareOperator::NotEq,
    CompareOperator::Lt,
    CompareOperator::Lte,
    CompareOperator::Gt,
    CompareOperator::Gte,
];

/// The result the array kernels must produce for `lhs op rhs`, derived only from [`Scalar`]'s
/// [`PartialEq`] and [`PartialOrd`] implementations.
fn expected_from_scalars(lhs: &Scalar, rhs: &Scalar, op: CompareOperator) -> Option<bool> {
    if lhs.is_null() || rhs.is_null() {
        return None;
    }

    let ordering = lhs
        .partial_cmp(rhs)
        .unwrap_or_else(|| panic!("{lhs} and {rhs} must be ordered"));
    // Equality and ordering must agree with each other before they can agree with the arrays.
    assert_eq!(
        lhs == rhs,
        ordering == Ordering::Equal,
        "PartialEq and PartialOrd disagree for {lhs} and {rhs}"
    );

    Some(match op {
        CompareOperator::Eq => lhs == rhs,
        CompareOperator::NotEq => lhs != rhs,
        CompareOperator::Lt => ordering.is_lt(),
        CompareOperator::Lte => ordering.is_le(),
        CompareOperator::Gt => ordering.is_gt(),
        CompareOperator::Gte => ordering.is_ge(),
    })
}

/// Build a canonical array holding `values` in order.
fn canonical_array(dtype: &DType, values: &[Scalar]) -> VortexResult<ArrayRef> {
    let mut builder = builder_with_capacity_in(dtype, values.len(), &SESSION.allocator());
    for value in values {
        builder.append_scalar(value)?;
    }
    Ok(builder.finish())
}

/// Execute `lhs op rhs` and check every row against [`expected_from_scalars`].
fn check_rows(
    lhs: ArrayRef,
    rhs: ArrayRef,
    lhs_values: &[&Scalar],
    rhs_values: &[&Scalar],
    op: CompareOperator,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = lhs.len();
    let result = lhs
        .binary(rhs, Operator::from(op))?
        .execute::<BoolArray>(&mut ctx)?;
    let validity = result
        .clone()
        .into_array()
        .validity()?
        .execute_mask(len, &mut ctx)?;
    let bits = result.into_bit_buffer();

    for (row, (lhs, rhs)) in lhs_values.iter().zip(rhs_values).enumerate() {
        let actual = validity.value(row).then(|| bits.value(row));
        let expected = expected_from_scalars(lhs, rhs, op);
        vortex_ensure!(
            actual == expected,
            "{lhs} {op} {rhs}: array compare returned {actual:?}, scalars give {expected:?}"
        );
    }

    Ok(())
}

/// Compare every ordered pair of `values` as canonical arrays and as array-vs-constant.
fn assert_array_scalar_parity(values: Vec<Scalar>) -> VortexResult<()> {
    let dtype = values[0].dtype().clone();
    let pairs: Vec<(&Scalar, &Scalar)> = values
        .iter()
        .flat_map(|lhs| values.iter().map(move |rhs| (lhs, rhs)))
        .collect();
    let (lhs_values, rhs_values): (Vec<_>, Vec<_>) = pairs.iter().copied().unzip();
    let lhs = canonical_array(
        &dtype,
        &lhs_values.iter().copied().cloned().collect::<Vec<_>>(),
    )?;
    let rhs = canonical_array(
        &dtype,
        &rhs_values.iter().copied().cloned().collect::<Vec<_>>(),
    )?;

    for op in OPERATORS {
        check_rows(lhs.clone(), rhs.clone(), &lhs_values, &rhs_values, op)?;
    }

    // Each value as a constant on either side of the full canonical array.
    let array = canonical_array(&dtype, &values)?;
    let all: Vec<&Scalar> = values.iter().collect();
    for constant in &values {
        let constant_array = ConstantArray::new(constant.clone(), values.len()).into_array();
        let constants = vec![constant; values.len()];
        for op in OPERATORS {
            check_rows(array.clone(), constant_array.clone(), &all, &constants, op)?;
            check_rows(constant_array.clone(), array.clone(), &constants, &all, op)?;
        }
    }

    Ok(())
}

fn bools() -> Vec<Scalar> {
    vec![
        Scalar::null(DType::Bool(Nullable)),
        Scalar::bool(false, Nullable),
        Scalar::bool(true, Nullable),
    ]
}

fn ints<T>(values: impl IntoIterator<Item = T>) -> Vec<Scalar>
where
    T: crate::dtype::NativePType + Into<crate::scalar::PValue>,
{
    std::iter::once(Scalar::null(DType::Primitive(T::PTYPE, Nullable)))
        .chain(
            values
                .into_iter()
                .map(|value| Scalar::primitive(value, Nullable)),
        )
        .collect()
}

/// Signed zeros, infinities, and NaNs with different signs and payloads.
fn f64s() -> Vec<Scalar> {
    ints([
        f64::NEG_INFINITY,
        -1.5,
        -0.0,
        0.0,
        f64::MIN_POSITIVE,
        1.5,
        f64::INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(f64::NAN.to_bits() | 1),
    ])
}

fn f32s() -> Vec<Scalar> {
    ints([
        f32::NEG_INFINITY,
        -1.5,
        -0.0,
        0.0,
        1.5,
        f32::INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(f32::NAN.to_bits() | 1),
    ])
}

fn f16s() -> Vec<Scalar> {
    ints([
        f16::NEG_INFINITY,
        f16::from_f32(-1.5),
        f16::NEG_ZERO,
        f16::ZERO,
        f16::from_f32(1.5),
        f16::INFINITY,
        f16::NAN,
        -f16::NAN,
    ])
}

fn decimals(
    precision: u8,
    scale: i8,
    values: impl IntoIterator<Item = DecimalValue>,
) -> Vec<Scalar> {
    let decimal_dtype = DecimalDType::new(precision, scale);
    std::iter::once(Scalar::null(DType::Decimal(decimal_dtype, Nullable)))
        .chain(
            values
                .into_iter()
                .map(|value| Scalar::decimal(value, decimal_dtype, Nullable)),
        )
        .collect()
}

/// Strings exercising empty values, embedded NULs, shared prefixes, and the 12-byte inline limit
/// of binary views.
const STRINGS: [&str; 10] = [
    "",
    "a",
    "a\0",
    "ab",
    "b",
    "é",
    "abcdefghijkl",
    "abcdefghijklm",
    "abcdefghijklmnopqrstuvwxyz",
    "abcdefghijklmnopqrstuvwxz",
];

fn utf8s() -> Vec<Scalar> {
    std::iter::once(Scalar::null(DType::Utf8(Nullable)))
        .chain(STRINGS.iter().map(|value| Scalar::utf8(*value, Nullable)))
        .collect()
}

fn binaries() -> Vec<Scalar> {
    let values: [&[u8]; 7] = [b"", b"\0", b"\0\0", b"\x7f", b"\x80", b"\xff", &[0xab; 20]];
    std::iter::once(Scalar::null(DType::Binary(Nullable)))
        .chain(
            values
                .iter()
                .map(|value| Scalar::binary(value.to_vec(), Nullable)),
        )
        .collect()
}

fn nullable_i32(value: Option<i32>) -> Scalar {
    value.map_or_else(
        || Scalar::null(DType::Primitive(PType::I32, Nullable)),
        |value| Scalar::primitive(value, Nullable),
    )
}

fn nullable_f64(value: Option<f64>) -> Scalar {
    value.map_or_else(
        || Scalar::null(DType::Primitive(PType::F64, Nullable)),
        |value| Scalar::primitive(value, Nullable),
    )
}

fn nullable_utf8(value: Option<&str>) -> Scalar {
    value.map_or_else(
        || Scalar::null(DType::Utf8(Nullable)),
        |value| Scalar::utf8(value, Nullable),
    )
}

/// Structs with nullable fields, so field-level nulls order inside the struct.
fn structs() -> Vec<Scalar> {
    let dtype = DType::Struct(
        StructFields::new(
            FieldNames::from(["a", "b"]),
            vec![
                DType::Primitive(PType::I32, Nullable),
                DType::Utf8(Nullable),
            ],
        ),
        Nullable,
    );
    let rows = [
        (None, None),
        (None, Some("a")),
        (Some(1), None),
        (Some(1), Some("a")),
        (Some(1), Some("b")),
        (Some(2), Some("a")),
        (Some(-1), Some("z")),
    ];
    std::iter::once(Scalar::null(dtype.clone()))
        .chain(
            rows.into_iter()
                .map(|(a, b)| Scalar::struct_(dtype.clone(), [nullable_i32(a), nullable_utf8(b)])),
        )
        .collect()
}

/// Lists of nullable floats, covering prefixes, element-level nulls, and float edge cases.
fn lists() -> Vec<Scalar> {
    let element = Arc::new(DType::Primitive(PType::F64, Nullable));
    let rows: [&[Option<f64>]; 11] = [
        &[],
        &[None],
        &[None, None],
        &[Some(1.0)],
        &[Some(1.0), None],
        &[Some(1.0), Some(2.0)],
        &[Some(1.0), Some(2.0), Some(3.0)],
        &[Some(2.0)],
        &[Some(-0.0)],
        &[Some(0.0)],
        &[Some(f64::NAN)],
    ];
    std::iter::once(Scalar::null(DType::List(Arc::clone(&element), Nullable)))
        .chain(rows.into_iter().map(|row| {
            Scalar::list(
                Arc::clone(&element),
                row.iter().copied().map(nullable_f64).collect(),
                Nullable,
            )
        }))
        .collect()
}

fn fixed_size_lists() -> Vec<Scalar> {
    let element = Arc::new(DType::Primitive(PType::I32, Nullable));
    let rows = [
        [None, None],
        [None, Some(1)],
        [Some(1), None],
        [Some(1), Some(1)],
        [Some(1), Some(2)],
        [Some(2), Some(0)],
    ];
    std::iter::once(Scalar::null(DType::FixedSizeList(
        Arc::clone(&element),
        2,
        Nullable,
    )))
    .chain(rows.into_iter().map(|row| {
        Scalar::fixed_size_list(
            Arc::clone(&element),
            row.into_iter().map(nullable_i32).collect(),
            Nullable,
        )
    }))
    .collect()
}

fn maps() -> VortexResult<Vec<Scalar>> {
    let dtype = DType::map(
        DType::Primitive(PType::I32, NonNullable),
        DType::Utf8(Nullable),
        false,
        Nullable,
    )?;
    let rows: [&[(i32, Option<&str>)]; 6] = [
        &[],
        &[(1, None)],
        &[(1, Some("a"))],
        &[(1, Some("b"))],
        &[(1, Some("a")), (2, Some("b"))],
        &[(2, Some("a"))],
    ];
    let mut values = vec![Scalar::null(dtype.clone())];
    for row in rows {
        values.push(Scalar::try_map(
            dtype.clone(),
            row.iter()
                .map(|(key, value)| (Scalar::primitive(*key, NonNullable), nullable_utf8(*value))),
        )?);
    }
    Ok(values)
}

fn timestamps() -> Vec<Scalar> {
    let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullable).erased();
    std::iter::once(Scalar::null(DType::Extension(ext_dtype.clone())))
        .chain([i64::MIN, -1, 0, 1, i64::MAX].into_iter().map(|millis| {
            Scalar::extension_ref(ext_dtype.clone(), Scalar::primitive(millis, Nullable))
        }))
        .collect()
}

#[rstest]
#[case::bool(bools())]
#[case::u8(ints([0u8, 1, 127, 128, u8::MAX]))]
#[case::u64(ints([0u64, 1, u64::MAX / 2, u64::MAX]))]
#[case::i8(ints([i8::MIN, -1, 0, 1, i8::MAX]))]
#[case::i32(ints([i32::MIN, -1, 0, 1, i32::MAX]))]
#[case::i64(ints([i64::MIN, -1, 0, 1, i64::MAX]))]
#[case::f16(f16s())]
#[case::f32(f32s())]
#[case::f64(f64s())]
#[case::decimal_i32(decimals(9, 2, [
    DecimalValue::I32(-999_999_999),
    DecimalValue::I32(-1),
    DecimalValue::I32(0),
    DecimalValue::I32(1),
    DecimalValue::I32(999_999_999),
]))]
#[case::decimal_i128(decimals(38, 0, [
    DecimalValue::I128(-(10i128.pow(38) - 1)),
    DecimalValue::I128(-1),
    DecimalValue::I128(0),
    DecimalValue::I128(i64::MAX as i128 + 1),
    DecimalValue::I128(10i128.pow(38) - 1),
]))]
#[case::utf8(utf8s())]
#[case::binary(binaries())]
#[case::struct_(structs())]
#[case::list(lists())]
#[case::fixed_size_list(fixed_size_lists())]
#[case::extension(timestamps())]
fn scalar_and_array_comparisons_agree(#[case] values: Vec<Scalar>) -> VortexResult<()> {
    assert_array_scalar_parity(values)
}

#[test]
fn map_scalar_and_array_comparisons_agree() -> VortexResult<()> {
    assert_array_scalar_parity(maps()?)
}

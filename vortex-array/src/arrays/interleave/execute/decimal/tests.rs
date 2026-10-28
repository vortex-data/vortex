// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::InterleaveArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::SliceArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::assert_arrays_eq;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// Executes the gather, checks its physical width and dtype, and compares it with a canonical oracle.
#[track_caller]
fn assert_interleaved(
    values: Vec<ArrayRef>,
    indices: &[(u8, u64)],
    expected: DecimalArray,
) -> VortexResult<()> {
    let branches = PrimitiveArray::from_iter(indices.iter().map(|&(branch, _)| branch));
    let rows = PrimitiveArray::from_iter(indices.iter().map(|&(_, row)| row));
    let array = InterleaveArray::try_new(values, branches.into_array(), rows.into_array())?;
    let mut ctx = SESSION.create_execution_ctx();
    let output = array.into_array().execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(
        output.values_type(),
        expected.decimal_dtype().canonical_values_type()
    );
    assert_eq!(output.dtype(), expected.dtype());
    assert_arrays_eq!(output.into_array(), expected.into_array(), &mut ctx);

    Ok(())
}

#[rstest]
#[case::i8(2)]
#[case::i16(4)]
#[case::i32(9)]
#[case::i64(18)]
#[case::i128(38)]
#[case::i256(76)]
fn test_decimal_interleave(
    #[case] precision: u8,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
    #[values(-3, 0, 2)] scale: i8,
) -> VortexResult<()> {
    let dtype = DecimalDType::new(precision, scale);
    let (first_validity, output_validity) = if nullability.is_nullable() {
        (
            Validity::from_iter([true, false, true]),
            Validity::from_iter([true, true, false, true, true, true]),
        )
    } else {
        (Validity::NonNullable, Validity::NonNullable)
    };
    match_each_decimal_value_type!(dtype.canonical_values_type(), |T| {
        let min = T::MIN_BY_PRECISION[usize::from(precision)];
        let max = T::MAX_BY_PRECISION[usize::from(precision)];
        let zero = T::default();
        let middle = if nullability.is_nullable() {
            T::MIN
        } else {
            zero
        };
        let first = DecimalArray::new(Buffer::copy_from([min, middle, max]), dtype, first_validity);
        let second = DecimalArray::new(
            Buffer::copy_from([max, min, zero]),
            dtype,
            Validity::NonNullable,
        );
        let expected = DecimalArray::new(
            Buffer::copy_from([zero, min, zero, max, max, min]),
            dtype,
            output_validity,
        );
        assert_interleaved(
            vec![first.into_array(), second.into_array()],
            &[(1, 2), (0, 0), (0, 1), (1, 0), (0, 2), (0, 0)],
            expected,
        )
    })
}

#[rstest]
#[case::i8(2)]
#[case::i16(4)]
#[case::i32(9)]
#[case::i64(18)]
#[case::i128(38)]
#[case::i256(76)]
fn test_decimal_interleave_constants(
    #[case] precision: u8,
    #[values(false, true)] null_constant: bool,
    #[values(DecimalType::I8, DecimalType::I256)] scalar_storage: DecimalType,
) -> VortexResult<()> {
    let dtype = DecimalDType::new(precision, 2);
    let scalar = if null_constant {
        Scalar::null(DType::Decimal(dtype, Nullability::Nullable))
    } else {
        let payload = match_each_decimal_value_type!(scalar_storage, |S| {
            DecimalValue::from(
                <S as BigCast>::from(-7i8).vortex_expect("-7 fits every decimal type"),
            )
        });
        Scalar::decimal(payload, dtype, Nullability::NonNullable)
    };
    let constant = ConstantArray::new(scalar, 3).into_array();
    match_each_decimal_value_type!(dtype.canonical_values_type(), |T| {
        let seven = <T as BigCast>::from(7i8).vortex_expect("7 fits every decimal type");
        let minus_seven = <T as BigCast>::from(-7i8).vortex_expect("-7 fits every decimal type");
        let column = DecimalArray::new(Buffer::copy_from([seven]), dtype, Validity::NonNullable)
            .into_array();
        let expected = DecimalArray::new(
            Buffer::copy_from([minus_seven, seven, minus_seven, minus_seven]),
            dtype,
            if null_constant {
                Validity::from_iter([false, true, false, false])
            } else {
                Validity::NonNullable
            },
        );
        assert_interleaved(
            vec![constant, column],
            &[(0, 2), (1, 0), (0, 1), (0, 2)],
            expected,
        )
    })
}

#[rstest]
#[case::i8(2)]
#[case::i16(4)]
#[case::i32(9)]
#[case::i64(18)]
#[case::i128(38)]
#[case::i256(76)]
fn test_decimal_interleave_mixed_stored_widths(#[case] precision: u8) -> VortexResult<()> {
    let dtype = DecimalDType::new(precision, 0);
    let first =
        DecimalArray::new(Buffer::copy_from([1i8, -2]), dtype, Validity::NonNullable).into_array();
    let second = DecimalArray::new(
        Buffer::copy_from([i256::from_i128(3), i256::MIN, i256::MAX]),
        dtype,
        Validity::from_iter([true, false, false]),
    )
    .into_array();
    match_each_decimal_value_type!(dtype.canonical_values_type(), |T| {
        let values = [3i8, -2, 0, 1, 0].map(|v| {
            <T as BigCast>::from(v).vortex_expect("small test values fit every decimal type")
        });
        let expected = DecimalArray::new(
            Buffer::copy_from(values),
            dtype,
            Validity::from_iter([true, true, false, true, false]),
        );
        assert_interleaved(
            vec![first, second],
            &[(1, 0), (0, 1), (1, 1), (0, 0), (1, 2)],
            expected,
        )
    })
}

#[rstest]
#[case::empty(&[])]
#[case::all_null(&[(0, 0), (1, 1), (0, 1)])]
fn test_decimal_interleave_without_valid_values(#[case] indices: &[(u8, u64)]) -> VortexResult<()> {
    let dtype = DecimalDType::new(9, -2);
    let first = DecimalArray::new(
        Buffer::copy_from([i256::MIN, i256::MAX]),
        dtype,
        Validity::AllInvalid,
    )
    .into_array();
    let second = ConstantArray::new(
        Scalar::null(DType::Decimal(dtype, Nullability::Nullable)),
        2,
    )
    .into_array();
    let expected = DecimalArray::new(
        Buffer::<i32>::zeroed(indices.len()),
        dtype,
        Validity::AllInvalid,
    );
    assert_interleaved(vec![first, second], indices, expected)
}

#[test]
fn test_decimal_interleave_empty_sources() -> VortexResult<()> {
    let dtype = DecimalDType::new(9, 0);
    let empty = DecimalArray::new(Buffer::<i8>::empty(), dtype, Validity::NonNullable).into_array();
    let expected = DecimalArray::new(Buffer::<i32>::empty(), dtype, Validity::NonNullable);
    assert_interleaved(vec![empty.clone(), empty], &[], expected)
}

#[test]
fn test_decimal_interleave_executes_lazy_children() -> VortexResult<()> {
    let dtype = DecimalDType::new(2, 0);
    let values =
        DecimalArray::new(Buffer::copy_from([1i8, -2]), dtype, Validity::NonNullable).into_array();
    let lazy = SliceArray::try_new(values.clone(), 0..2)?.into_array();
    let branches = SliceArray::try_new(PrimitiveArray::from_iter([1u8, 0, 1]).into_array(), 0..3)?
        .into_array();
    let rows = SliceArray::try_new(PrimitiveArray::from_iter([1u16, 0, 0]).into_array(), 0..3)?
        .into_array();
    let interleaved = InterleaveArray::try_new(vec![values, lazy], branches, rows)?.into_array();
    let expected = DecimalArray::new(
        Buffer::copy_from([-2i8, 1, 1]),
        dtype,
        Validity::NonNullable,
    )
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(interleaved, expected, &mut ctx);

    Ok(())
}

#[rstest]
#[case::array_index(2, 0, "array index out of bounds")]
#[case::row_index(0, 2, "row index out of bounds")]
#[case::constant_row_index(1, 2, "row index out of bounds")]
#[case::large_array_index(u64::MAX, 0, "array index out of bounds")]
fn test_decimal_interleave_rejects_out_of_bounds_selectors(
    #[case] branch: u64,
    #[case] row: u8,
    #[case] message: &str,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let dtype = DecimalDType::new(2, 0);
    let values = DecimalArray::new(
        Buffer::copy_from([1i8, -2]),
        dtype,
        Validity::from(nullability),
    )
    .into_array();
    let constant =
        ConstantArray::new(Scalar::decimal(1i8.into(), dtype, nullability), 2).into_array();
    let array = InterleaveArray::try_new(
        vec![values, constant],
        PrimitiveArray::from_iter([branch]).into_array(),
        PrimitiveArray::from_iter([row]).into_array(),
    )?
    .into_array();
    let error = array
        .execute::<DecimalArray>(&mut SESSION.create_execution_ctx())
        .unwrap_err();
    assert!(error.to_string().contains(message), "{error}");

    Ok(())
}

#[test]
fn test_decimal_interleave_rejects_wide_values_outside_precision() -> VortexResult<()> {
    let dtype = DecimalDType::new(2, 0);
    let source =
        DecimalArray::new(Buffer::copy_from([100i16]), dtype, Validity::NonNullable).into_array();
    let array = InterleaveArray::try_new(
        vec![source.clone(), source],
        PrimitiveArray::from_iter([0u8]).into_array(),
        PrimitiveArray::from_iter([0u8]).into_array(),
    )?
    .into_array();
    let error = array
        .execute::<DecimalArray>(&mut SESSION.create_execution_ctx())
        .unwrap_err();
    assert!(error.to_string().contains("precision"), "{error}");

    Ok(())
}

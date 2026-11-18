// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stored integer conversion in both directions, without a copy of the stored child.

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::SESSION;
use super::widen;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::DecimalArray;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::validity::Validity;

/// A decimal `Widen`, whose values are not a stored integer.
fn decimal_widen() -> ArrayRef {
    let values = DecimalArray::new(
        buffer![1i16, 2],
        DecimalDType::new(4, 2),
        Validity::NonNullable,
    );
    let dtype = DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable);

    WidenArray::try_new(values.into_array(), dtype)
        .vortex_expect("the fixture child has a lower precision at the same scale")
        .into_array()
}

/// An integer `Widen` that is the values of a `Dict`, as a lazy take leaves it.
fn widen_below_dict() -> ArrayRef {
    let values = widen(buffer![1i8, 2], PType::I64).into_array();

    DictArray::try_new(buffer![1u8, 0].into_array(), values)
        .vortex_expect("the fixture codes index the values")
        .into_array()
}

/// The read step keeps the stored child, and the write step returns that same child.
#[rstest]
#[case::same_width(
    buffer![i64::MIN, 0, i64::MAX].into_array(),
    buffer![i64::MIN, 0, i64::MAX].into_array(),
)]
#[case::narrower_signed(
    buffer![i8::MIN, 0, i8::MAX].into_array(),
    buffer![-128i64, 0, 127].into_array(),
)]
#[case::narrower_unsigned(
    buffer![0u16, u16::MAX].into_array(),
    buffer![0u32, 65_535].into_array(),
)]
#[case::narrower_nullable(
    PrimitiveArray::from_option_iter([Some(i16::MIN), None, Some(i16::MAX)]).into_array(),
    PrimitiveArray::from_option_iter([Some(-32_768i32), None, Some(32_767)]).into_array(),
)]
fn test_stored_integer_round_trips_without_copy(
    #[case] stored: ArrayRef,
    #[case] expected: ArrayRef,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = expected.dtype().clone();

    let array = WidenArray::from_stored_integer(stored.clone(), dtype.clone())?;

    assert_eq!(array.dtype(), &dtype);
    assert_eq!(array.is::<Widen>(), stored.dtype() != &dtype);
    assert!(ArrayRef::ptr_eq(
        &WidenArray::to_stored_integer(&array),
        &stored
    ));
    assert_arrays_eq!(array, expected, &mut ctx);

    Ok(())
}

#[rstest]
#[case::wider_stored(PType::I64.into(), PType::I32.into(), "narrower")]
#[case::signedness(  PType::U8.into(),  PType::I64.into(), "signedness")]
#[case::decimal_dtype(
    DType::Decimal(DecimalDType::new(4, 2), Nullability::NonNullable),
    DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable),
    "integer dtype"
)]
fn test_from_stored_integer_rejects_invalid_dtypes(
    #[case] stored: DType,
    #[case] dtype: DType,
    #[case] requirement: &str,
) {
    let stored = Canonical::empty(&stored).into_array();

    let error = WidenArray::from_stored_integer(stored, dtype)
        .expect_err("the dtypes break a stored integer requirement");

    assert!(error.to_string().contains(requirement), "{error}");
}

/// Only an integer `Widen` at the root has a narrower stored form.
#[rstest]
#[case::primitive(buffer![1i64, 2].into_array())]
#[case::decimal_widen(decimal_widen())]
#[case::widen_below_dict(widen_below_dict())]
fn test_to_stored_integer_returns_other_arrays_unchanged(#[case] array: ArrayRef) {
    assert!(ArrayRef::ptr_eq(
        &WidenArray::to_stored_integer(&array),
        &array
    ));
}

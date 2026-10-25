// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::RangeInclusive;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::SESSION;
use super::stored_ptype;
use super::widen;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyArrayData;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::WidenSlots;
use crate::arrays::widen::encode::smallest_integer_ptype;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::optimizer::ArrayOptimizer;
use crate::validity::Validity;

#[rstest]
#[case::i8_bounds(   PrimitiveArray::from_iter([-128i64, 127]),                      PType::I8)]
#[case::i16_lower(   PrimitiveArray::from_iter([-129i64, 0]),                        PType::I16)]
#[case::i64_upper(   PrimitiveArray::from_iter([0, i64::from(i32::MAX) + 1]),        PType::I64)]
#[case::non_negative(PrimitiveArray::from_iter([0i64, 200]),                         PType::I16)]
#[case::u8_bounds(   PrimitiveArray::from_iter([0u64, 255]),                         PType::U8)]
#[case::u64(         PrimitiveArray::from_iter([u64::from(u32::MAX) + 1, u64::MAX]), PType::U64)]
#[case::i8_logical(  PrimitiveArray::from_iter([-128i8, 127]),                       PType::I8)]
#[case::empty(       PrimitiveArray::empty::<i64>(Nullability::NonNullable),         PType::I8)]
#[case::all_null(
    PrimitiveArray::new(buffer![i64::MIN, i64::MAX], Validity::AllInvalid),
    PType::I8
)]
#[case::null_payloads(
    PrimitiveArray::new(
        buffer![i64::MIN, 42, i64::MAX],
        Validity::from_iter([false, true, false])
    ),
    PType::I8
)]
fn test_encode_picks_smallest_width(
    #[case] input: PrimitiveArray,
    #[case] stored: PType,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = WidenArray::encode(input.clone(), &mut ctx)?;

    assert_eq!(stored_ptype(&encoded), stored);
    assert_eq!(encoded.dtype(), input.dtype());
    assert_arrays_eq!(encoded, input, &mut ctx);

    Ok(())
}

#[test]
fn test_encode_rejects_float() {
    let mut ctx = SESSION.create_execution_ctx();
    let error = WidenArray::encode(PrimitiveArray::from_iter([1.0f32]), &mut ctx)
        .expect_err("Widen stores only integers");

    assert!(error.to_string().contains("integer"), "{error}");
}

#[rstest]
#[case::non_negative( PrimitiveArray::from_iter([0i64, 200]),                 PType::I16)]
#[case::partially_valid(
    PrimitiveArray::from_option_iter([Some(0i64), None, Some(127)]),
    PType::I8
)]
#[case::u64_unchanged(PrimitiveArray::from_iter([0u64, u64::MAX]),            PType::U64)]
fn test_encode_values_keeps_signedness(
    #[case] input: PrimitiveArray,
    #[case] stored: PType,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    #[expect(deprecated, reason = "the test covers the deprecated function")]
    let values = WidenArray::encode_values(input.clone(), &mut ctx)?;

    assert_eq!(
        values.dtype(),
        &DType::Primitive(stored, input.dtype().nullability())
    );
    let widened = values.into_array().cast(input.dtype().clone())?;
    assert_arrays_eq!(widened, input, &mut ctx);

    Ok(())
}

#[test]
fn test_encode_values_preserves_floats() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = PrimitiveArray::from_iter([1.0f32, 2.0, 3.0]);
    #[expect(deprecated, reason = "the test covers the deprecated function")]
    let values = WidenArray::encode_values(input.clone(), &mut ctx)?;

    assert_eq!(values.dtype(), input.dtype());
    assert_arrays_eq!(values, input, &mut ctx);

    Ok(())
}

#[rstest]
#[case::signed_i8(        -128..=127,                   true,  Some(PType::I8))]
#[case::signed_i16(       -129..=0,                     true,  Some(PType::I16))]
#[case::below_i64(        i128::from(i64::MIN) - 1..=0, true,  None)]
#[case::unsigned_u8(      0..=255,                      false, Some(PType::U8))]
#[case::unsigned_u16(     0..=256,                      false, Some(PType::U16))]
#[case::above_u64(        0..=i128::from(u64::MAX) + 1, false, None)]
#[case::negative_unsigned(-1..=0,                       false, None)]
fn test_smallest_integer_ptype(
    #[case] bounds: RangeInclusive<i128>,
    #[case] signed: bool,
    #[case] expected: Option<PType>,
) {
    assert_eq!(smallest_integer_ptype(bounds, signed), expected);
}

/// Both `try_new` and `VTable::validate`, which `try_from_parts` runs, reject the dtypes.
#[rstest]
#[case::bool_values(  DType::Bool(Nullability::NonNullable), PType::I64.into(), "integer dtypes")]
#[case::float_values( PType::F32.into(),                     PType::I64.into(), "integer dtypes")]
#[case::signedness(   PType::I8.into(),                      PType::U64.into(), "signedness")]
#[case::equal_width(  PType::I8.into(),                      PType::I8.into(),  "narrower")]
#[case::nullability(
    PType::I8.into(),
    DType::Primitive(PType::I64, Nullability::Nullable),
    "nullability"
)]
fn test_rejects_invalid_dtypes(
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
        .expect_err("the dtypes break a Widen requirement")
        .to_string();
    let validate_error = WidenArray::try_from_parts(parts)
        .expect_err("the dtypes break a Widen requirement")
        .to_string();

    assert!(try_new_error.contains(requirement), "{try_new_error}");
    assert!(validate_error.contains(requirement), "{validate_error}");
}

#[test]
fn test_try_new_rejects_widen_child_at_logical_width() {
    let child = widen(buffer![1i8], PType::I32).into_array();
    let error = WidenArray::try_new(child, PType::I32.into())
        .expect_err("the `i32` child is not narrower than the logical dtype");

    assert!(error.to_string().contains("narrower"), "{error}");
}

#[rstest]
#[case::missing_values( vec![None],                                  1)]
#[case::length_mismatch(vec![Some(buffer![1i8].into_array())],       2)]
fn test_validate_rejects_malformed_slots(#[case] slots: Vec<Option<ArrayRef>>, #[case] len: usize) {
    let parts = ArrayParts::new(Widen, PType::I64.into(), len, EmptyArrayData, slots.into());

    assert!(WidenArray::try_from_parts(parts).is_err());
}

#[test]
fn test_try_new_flattens_widen_values() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_option_iter([Some(i8::MIN), None, Some(i8::MAX)]);
    let inner = WidenArray::try_new(
        values.into_array(),
        DType::Primitive(PType::I16, Nullability::Nullable),
    )?;
    let outer = WidenArray::try_new(
        inner.into_array(),
        DType::Primitive(PType::I64, Nullability::Nullable),
    )?;

    assert!(outer.values().is::<Primitive>());
    let expected = PrimitiveArray::from_option_iter([Some(-128i64), None, Some(127)]);
    assert_arrays_eq!(outer, expected, &mut ctx);

    Ok(())
}

#[test]
fn test_reduce_flattens_nested_widen() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values =
        PrimitiveArray::from_option_iter([Some(i8::MIN), None, Some(i8::MAX)]).into_array();
    let inner = widen(values.clone(), PType::I16).into_array();
    let nested = WidenArray::try_from_parts(ArrayParts::new(
        Widen,
        DType::Primitive(PType::I64, Nullability::Nullable),
        inner.len(),
        EmptyArrayData,
        WidenSlots { values: inner }.into_slots(),
    ))?;
    assert!(nested.values().is::<Widen>());

    let flattened = nested.into_array().optimize(&SESSION)?;

    assert!(ArrayRef::ptr_eq(flattened.as_::<Widen>().values(), &values));
    let expected = PrimitiveArray::from_option_iter([Some(-128i64), None, Some(127)]);
    assert_arrays_eq!(flattened, expected, &mut ctx);

    Ok(())
}

/// One execution step yields a cast of the values, so the widening moves into the values of the
/// `Dict` and its codes stay as they are.
#[test]
fn test_execute_encoded_values() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let codes = buffer![2u8, 0, 1, 2].into_array();
    let values = DictArray::try_new(codes.clone(), buffer![-1i8, 0, 1].into_array())?;
    let array = WidenArray::try_new(values.into_array(), PType::I64.into())?;
    assert!(array.values().is::<Dict>());

    // The step yields a lazy cast of the values, which optimizing pushes into the `Dict`.
    let step = array
        .clone()
        .into_array()
        .execute::<ArrayRef>(&mut ctx)?
        .optimize(&SESSION)?;

    assert!(step.is::<Dict>());
    assert!(ArrayRef::ptr_eq(step.as_::<Dict>().codes(), &codes));
    assert_eq!(step.dtype(), &DType::from(PType::I64));
    assert_arrays_eq!(array, PrimitiveArray::from_iter([1i64, -1, 0, 1]), &mut ctx);

    Ok(())
}

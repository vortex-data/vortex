// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arithmetic matches canonical integers and keeps sufficient stored widths.

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::Narrow;
use crate::arrays::NarrowArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::narrow::NarrowArraySlotsExt;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

#[rstest]
#[case::same_width(vec![1, -2], vec![2, 3], Operator::Add, PType::I8)]
#[case::grow_add(vec![127, -128], vec![1, -1], Operator::Add, PType::I16)]
#[case::grow_sub(vec![-128, 127], vec![1, -1], Operator::Sub, PType::I16)]
#[case::skip_width(vec![32767, -32768], vec![32767, 2], Operator::Mul, PType::I32)]
#[case::full_width(vec![i64::from(i32::MIN), 2], vec![-1, 3], Operator::Mul, PType::I64)]
#[case::divide_min(vec![-128, 127], vec![-1, 2], Operator::Div, PType::I16)]
#[case::divide_cross_zero(vec![-128, 127], vec![2, -2], Operator::Div, PType::I16)]
#[case::overestimated_interval(vec![0, 1], vec![i64::MAX, 0], Operator::Add, PType::I64)]
fn signed_arithmetic(
    #[case] lhs: Vec<i64>,
    #[case] rhs: Vec<i64>,
    #[case] operator: Operator,
    #[case] storage: PType,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = PrimitiveArray::from_iter(lhs);
    let rhs = PrimitiveArray::from_iter(rhs);
    let encoded_lhs = NarrowArray::encode(lhs.clone(), &mut ctx)?;
    let encoded_rhs = NarrowArray::encode(rhs.clone(), &mut ctx)?;
    let result = encoded_lhs
        .binary(encoded_rhs, operator)?
        .execute::<ArrayRef>(&mut ctx)?;
    let actual_storage = result.as_opt::<Narrow>().map_or_else(
        || result.dtype().as_ptype(),
        |array| array.values().dtype().as_ptype(),
    );

    assert_eq!(actual_storage, storage);
    assert_eq!(result.dtype(), lhs.dtype());
    assert_arrays_eq!(
        result,
        lhs.into_array().binary(rhs.into_array(), operator)?,
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case::add(Operator::Add)]
#[case::subtract(Operator::Sub)]
#[case::multiply(Operator::Mul)]
#[case::divide(Operator::Div)]
fn signed_domain(#[case] operator: Operator) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = PrimitiveArray::from_iter((-128i64..=127).flat_map(|value| [value; 4]));
    let rhs = PrimitiveArray::from_iter([-2i64, -1, 1, 2].into_iter().cycle().take(1024));
    let encoded_lhs = NarrowArray::encode(lhs.clone(), &mut ctx)?;
    let encoded_rhs = NarrowArray::encode(rhs.clone(), &mut ctx)?;

    assert_arrays_eq!(
        encoded_lhs.binary(encoded_rhs, operator)?,
        lhs.into_array().binary(rhs.into_array(), operator)?,
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case::add(vec![255u64, 0], vec![1, 2], Operator::Add)]
#[case::multiply(vec![u64::from(u32::MAX), 2], vec![u64::from(u32::MAX), 3], Operator::Mul)]
#[case::subtract_interval(vec![0u64, 1], vec![0, 1], Operator::Sub)]
#[case::divide(vec![255u64, 200], vec![2, 3], Operator::Div)]
fn unsigned_arithmetic(
    #[case] lhs: Vec<u64>,
    #[case] rhs: Vec<u64>,
    #[case] operator: Operator,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = PrimitiveArray::from_iter(lhs);
    let rhs = PrimitiveArray::from_iter(rhs);
    let encoded_lhs = NarrowArray::encode(lhs.clone(), &mut ctx)?;
    let encoded_rhs = NarrowArray::encode(rhs.clone(), &mut ctx)?;

    assert_arrays_eq!(
        encoded_lhs.binary(encoded_rhs, operator)?,
        lhs.into_array().binary(rhs.into_array(), operator)?,
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case::add(Operator::Add, i64::MAX)]
#[case::subtract(Operator::Sub, i64::MIN)]
#[case::multiply(Operator::Mul, i64::MAX)]
#[case::zero_divisor(Operator::Div, 0)]
fn logical_errors(#[case] operator: Operator, #[case] rhs: i64) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = PrimitiveArray::from_iter([2i64, 3]);
    let encoded = NarrowArray::encode(input.clone(), &mut ctx)?;
    let rhs = ConstantArray::new(rhs, input.len()).into_array();

    assert!(
        encoded
            .binary(rhs.clone(), operator)?
            .execute::<PrimitiveArray>(&mut ctx)
            .is_err()
    );
    assert!(
        input
            .into_array()
            .binary(rhs, operator)?
            .execute::<PrimitiveArray>(&mut ctx)
            .is_err()
    );

    Ok(())
}

#[test]
fn unsigned_underflow() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = NarrowArray::try_new(buffer![0u8, 1].into_array(), PType::U64.into())?.into_array();
    let rhs = ConstantArray::new(1u64, 2).into_array();
    assert!(
        array
            .binary(rhs, Operator::Sub)?
            .execute::<PrimitiveArray>(&mut ctx)
            .is_err()
    );

    Ok(())
}

#[rstest]
#[case::empty(PrimitiveArray::new(Buffer::<i8>::empty(), Validity::NonNullable))]
#[case::all_null(PrimitiveArray::new(buffer![-128i8, 127], Validity::AllInvalid))]
#[case::null_payload_overflow(PrimitiveArray::new(buffer![-128i8, 127], Validity::from_iter([false, true])))]
fn nulls_and_empty(#[case] values: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DType::Primitive(PType::I64, values.dtype().nullability());
    let input = values.clone().into_array().cast(dtype.clone())?;
    let array = NarrowArray::try_new(values.into_array(), dtype)?.into_array();
    let rhs = ConstantArray::new(-1i64, input.len()).into_array();

    assert_arrays_eq!(
        array.binary(rhs.clone(), Operator::Div)?,
        input.binary(rhs, Operator::Div)?,
        &mut ctx
    );

    Ok(())
}

#[test]
fn encoded_child_and_operand_order() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let child = DictArray::try_new(
        buffer![0u8, 1, 0].into_array(),
        buffer![-128i8, 127].into_array(),
    )?
    .into_array();
    let array = NarrowArray::try_new(child, PType::I64.into())?.into_array();
    let rhs = ConstantArray::new(1000i64, 3).into_array();

    assert_arrays_eq!(
        rhs.binary(array.clone(), Operator::Sub)?,
        buffer![1128i64, 873, 1128].into_array(),
        &mut ctx
    );
    assert_arrays_eq!(array, buffer![-128i64, 127, -128].into_array(), &mut ctx);

    Ok(())
}

#[rstest]
#[case::i16(PType::I16, i64::from(i16::MAX))]
#[case::i32(PType::I32, i64::from(i32::MAX))]
#[case::i64(PType::I64, i64::MAX)]
fn overflow_at_each_logical_width(#[case] logical: PType, #[case] max: i64) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = NarrowArray::try_new(buffer![2i8, 3].into_array(), logical.into())?.into_array();
    let rhs = ConstantArray::new(Scalar::from(max).cast(&logical.into())?, 2).into_array();
    assert!(
        array
            .binary(rhs, Operator::Mul)?
            .execute::<PrimitiveArray>(&mut ctx)
            .is_err()
    );

    Ok(())
}

#[test]
fn logical_overflow_behind_null() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let child =
        PrimitiveArray::new(buffer![2i8, 3], Validity::from_iter([false, true])).into_array();
    let array = NarrowArray::try_new(child, DType::Primitive(PType::I64, Nullability::Nullable))?
        .into_array();

    assert_arrays_eq!(
        array.binary(buffer![i64::MAX, 2].into_array(), Operator::Mul)?,
        PrimitiveArray::from_option_iter([None, Some(6i64)]),
        &mut ctx
    );

    Ok(())
}

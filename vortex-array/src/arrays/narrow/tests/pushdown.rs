// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selection and predicates retain the logical dtype and narrow children.

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::Interleave;
use crate::arrays::InterleaveArray;
use crate::arrays::Narrow;
use crate::arrays::NarrowArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::interleave::InterleaveArrayExt;
use crate::arrays::narrow::NarrowArraySlotsExt;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;

#[rstest]
#[case::same_width(2i64, PType::I8)]
#[case::next_width(1000i64, PType::I16)]
#[case::full_width(i64::MAX, PType::I64)]
fn fill_width(#[case] fill: i64, #[case] storage: PType) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = PrimitiveArray::from_option_iter([Some(1i64), None, Some(-2)]);
    let array = NarrowArray::encode(input, &mut ctx)?;
    let filled = array.fill_null(fill)?.optimize()?;
    let actual_storage = filled.as_opt::<Narrow>().map_or_else(
        || filled.dtype().as_ptype(),
        |array| array.values().dtype().as_ptype(),
    );

    assert_eq!(actual_storage, storage);
    assert_arrays_eq!(filled, buffer![1i64, fill, -2].into_array(), &mut ctx);

    Ok(())
}

#[rstest]
#[case::constant(false)]
#[case::other_narrow(true)]
fn zip_width(#[case] other_narrow: bool) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = PrimitiveArray::from_option_iter([Some(-128i64), None, Some(127)]);
    let array = NarrowArray::encode(input.clone(), &mut ctx)?;
    let other = if other_narrow {
        NarrowArray::try_new(buffer![1000i16, 2, 300].into_array(), PType::I64.into())?.into_array()
    } else {
        ConstantArray::new(1000i64, 3).into_array()
    };
    let mask = BoolArray::from_iter([true, false, true]).into_array();
    let result = mask.zip(array, other.clone())?.optimize()?;

    assert_eq!(
        result.as_::<Narrow>().values().dtype().as_ptype(),
        PType::I16
    );
    assert_arrays_eq!(result, mask.zip(input.into_array(), other)?, &mut ctx);

    Ok(())
}

#[rstest]
fn between_bounds(
    #[values(-129i64, -128, 0, 128)] lower: i64,
    #[values(-128i64, 0, 127, 128)] upper: i64,
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] lower_strict: StrictComparison,
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] upper_strict: StrictComparison,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input =
        PrimitiveArray::from_option_iter([Some(-128i64), None, Some(0), Some(127)]).into_array();
    let array = NarrowArray::encode(input.clone().execute::<PrimitiveArray>(&mut ctx)?, &mut ctx)?;
    let lower = ConstantArray::new(lower, input.len()).into_array();
    let upper = ConstantArray::new(upper, input.len()).into_array();
    let options = BetweenOptions {
        lower_strict,
        upper_strict,
    };

    assert_arrays_eq!(
        array.between(lower.clone(), upper.clone(), options.clone())?,
        input.between(lower, upper, options)?,
        &mut ctx
    );

    Ok(())
}

#[test]
fn between_nullable_bounds() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = buffer![-128i64, 0, 127].into_array();
    let array = NarrowArray::encode(input.clone().execute::<PrimitiveArray>(&mut ctx)?, &mut ctx)?;
    let lower = PrimitiveArray::from_option_iter([Some(-129i64), None, Some(128)]).into_array();
    let upper = ConstantArray::new(0i64, 3).into_array();
    let options = BetweenOptions {
        lower_strict: StrictComparison::NonStrict,
        upper_strict: StrictComparison::NonStrict,
    };

    assert_arrays_eq!(
        array.between(lower.clone(), upper.clone(), options.clone())?,
        input.between(lower, upper, options)?,
        &mut ctx
    );

    Ok(())
}

#[test]
fn zip_false_branch_and_null_mask() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = buffer![-128i64, 0, 127].into_array();
    let array = NarrowArray::encode(input.execute::<PrimitiveArray>(&mut ctx)?, &mut ctx)?;
    let other = ConstantArray::new(1000i64, 3).into_array();
    let mask = BoolArray::from_iter([Some(true), None, Some(false)]).into_array();
    let result = mask.zip(other, array)?.optimize()?;

    assert_eq!(
        result.as_::<Narrow>().values().dtype().as_ptype(),
        PType::I16
    );
    assert_arrays_eq!(result, buffer![1000i64, 0, 127].into_array(), &mut ctx);

    Ok(())
}

#[test]
fn narrowed_take_indices() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let codes =
        NarrowArray::try_new(buffer![2u8, 0, 1].into_array(), PType::U64.into())?.into_array();
    let array = DictArray::try_new(codes, buffer![10i64, 20, 30].into_array())?
        .into_array()
        .optimize()?;

    assert_arrays_eq!(array, buffer![30i64, 10, 20].into_array(), &mut ctx);

    Ok(())
}

#[rstest]
#[case::unsigned_to_signed(buffer![0u8, 255].into_array(), PType::U64, PType::I32, Some(PType::I16))]
#[case::signed_to_unsigned(buffer![0i8, 127].into_array(), PType::I64, PType::U32, Some(PType::U8))]
#[case::float(buffer![-128i8, 127].into_array(), PType::I64, PType::F64, None)]
fn cast_child(
    #[case] values: crate::ArrayRef,
    #[case] logical: PType,
    #[case] target: PType,
    #[case] storage: Option<PType>,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = values.cast(logical.into())?;
    let array = NarrowArray::try_new(values, logical.into())?.into_array();
    let result = array.cast(target.into())?.optimize()?;
    if let Some(storage) = storage {
        assert_eq!(result.as_::<Narrow>().values().dtype().as_ptype(), storage);
    }
    assert_arrays_eq!(result, input.cast(target.into())?, &mut ctx);

    Ok(())
}

#[test]
fn signedness_cast_checks_values_and_nulls() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values = PrimitiveArray::from_option_iter([Some(-1i8), None]).into_array();
    let array = NarrowArray::try_new(values, DType::Primitive(PType::I64, Nullability::Nullable))?
        .into_array();
    assert!(
        array
            .cast(DType::Primitive(PType::U64, Nullability::Nullable))?
            .execute::<PrimitiveArray>(&mut ctx)
            .is_err()
    );

    let values = PrimitiveArray::from_option_iter([Some(1i8), None]).into_array();
    let array = NarrowArray::try_new(values, DType::Primitive(PType::I64, Nullability::Nullable))?
        .into_array();
    assert!(
        array
            .cast(PType::U64.into())?
            .execute::<PrimitiveArray>(&mut ctx)
            .is_err()
    );

    Ok(())
}

#[test]
fn concat_stored_widths() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let first =
        NarrowArray::try_new(buffer![-128i8, 127].into_array(), PType::I64.into())?.into_array();
    let second =
        NarrowArray::try_new(buffer![1000i16, 2000].into_array(), PType::I64.into())?.into_array();
    let result = ChunkedArray::try_new([first, second], PType::I64.into())?
        .into_array()
        .optimize()?;

    assert_eq!(
        result.as_::<Narrow>().values().dtype().as_ptype(),
        PType::I16
    );
    assert_arrays_eq!(
        result,
        buffer![-128i64, 127, 1000, 2000].into_array(),
        &mut ctx
    );

    Ok(())
}

#[test]
fn interleave_stored_widths() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let first =
        NarrowArray::try_new(buffer![-128i8, 127].into_array(), PType::I64.into())?.into_array();
    let second =
        NarrowArray::try_new(buffer![1000i16, 2000].into_array(), PType::I64.into())?.into_array();
    let result = InterleaveArray::try_new(
        vec![first, second],
        buffer![1u8, 0, 1, 0].into_array(),
        buffer![0u8, 1, 1, 0].into_array(),
    )?
    .into_array()
    .optimize()?;

    assert_eq!(
        result.as_::<Narrow>().values().dtype().as_ptype(),
        PType::I16
    );
    assert_arrays_eq!(
        result,
        buffer![1000i64, 127, 2000, -128].into_array(),
        &mut ctx
    );

    Ok(())
}

#[test]
fn interleave_stored_selectors() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let arrays =
        NarrowArray::try_new(buffer![0u8, 1, 0].into_array(), PType::U64.into())?.into_array();
    let rows =
        NarrowArray::try_new(buffer![1u8, 0, 0].into_array(), PType::U64.into())?.into_array();
    let result = InterleaveArray::try_new(
        vec![
            buffer![10i64, 20].into_array(),
            buffer![30i64, 40].into_array(),
        ],
        arrays,
        rows,
    )?
    .into_array()
    .optimize()?;

    assert_eq!(
        result
            .as_::<Interleave>()
            .array_indices()
            .dtype()
            .as_ptype(),
        PType::U8
    );
    assert_eq!(
        result.as_::<Interleave>().row_indices().dtype().as_ptype(),
        PType::U8
    );
    assert_arrays_eq!(result, buffer![20i64, 30, 10].into_array(), &mut ctx);

    Ok(())
}

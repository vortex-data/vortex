// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::nullable_union_array;
use super::nullable_variants;
use super::union_array;
use super::variants;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::PrimitiveArray;
use crate::arrays::Union;
use crate::arrays::UnionArray;
use crate::arrays::dict::TakeReduce;
use crate::arrays::union::UnionArrayExt;
use crate::compute::conformance::take::test_take_conformance;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;

/// Take `indices`, asserting the result reduced back to a union instead of staying a dictionary.
#[track_caller]
fn take(array: &UnionArray, indices: ArrayRef) -> VortexResult<UnionArray> {
    Ok(array
        .clone()
        .into_array()
        .take(indices)?
        .as_::<Union>()
        .into_owned())
}

/// Assert that `array` holds exactly `expected`, row for row.
#[track_caller]
fn assert_rows(array: &UnionArray, expected: Vec<Scalar>) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(array.len(), expected.len());
    for (index, expected) in expected.into_iter().enumerate() {
        assert_eq!(array.execute_scalar(index, &mut ctx)?, expected);
    }

    Ok(())
}

#[test]
fn take_reorders_and_repeats_variant_selection() -> VortexResult<()> {
    let variants = variants()?;
    let nullability = Nullability::NonNullable;

    let taken = take(
        &union_array()?,
        PrimitiveArray::from_iter([2u64, 1, 0, 1]).into_array(),
    )?;

    assert_eq!(taken.dtype(), &DType::Union(variants.clone(), nullability));
    assert_rows(
        &taken,
        vec![
            Scalar::union(variants.clone(), 5, 30i32.into(), nullability)?, //
            Scalar::union(variants.clone(), 9, true.into(), nullability)?,  //
            Scalar::union(variants.clone(), 5, 10i32.into(), nullability)?, //
            Scalar::union(variants, 9, true.into(), nullability)?,          //
        ],
    )
}

#[test]
fn null_indices_become_outer_nulls_and_leave_children_alone() -> VortexResult<()> {
    let variants = variants()?;
    let nullability = Nullability::Nullable;

    let taken = take(
        &union_array()?,
        PrimitiveArray::from_option_iter([Some(1u64), None, Some(0)]).into_array(),
    )?;

    assert_eq!(taken.dtype(), &DType::Union(variants.clone(), nullability));

    // A nullable index widens the union but never its variants.
    assert_eq!(
        taken.child_by_name("number")?.dtype(),
        &DType::Primitive(PType::I32, Nullability::NonNullable)
    );
    assert_eq!(
        taken.child_by_name("flag")?.dtype(),
        &DType::Bool(Nullability::NonNullable)
    );

    assert_rows(
        &taken,
        vec![
            Scalar::union(variants.clone(), 9, true.into(), nullability)?, //
            Scalar::null(DType::Union(variants.clone(), nullability)),     //
            Scalar::union(variants, 5, 10i32.into(), nullability)?,        //
        ],
    )
}

#[test]
fn take_keeps_outer_and_inner_nulls_distinct() -> VortexResult<()> {
    let variants = nullable_variants()?;
    let nullability = Nullability::Nullable;

    // Row 1 is an outer null and row 2 is a present union selecting a null child.
    let taken = take(
        &nullable_union_array()?,
        PrimitiveArray::from_iter([1u64, 2, 3]).into_array(),
    )?;

    assert_rows(
        &taken,
        vec![
            Scalar::null(DType::Union(variants.clone(), nullability)), //
            Scalar::union(
                variants.clone(),
                9,
                Scalar::null(DType::Primitive(PType::I64, nullability)),
                nullability,
            )?, //
            Scalar::union(
                variants,
                9,
                Scalar::primitive(40i64, nullability),
                nullability,
            )?, //
        ],
    )
}

/// `take` short-circuits an empty source into a constant before the union sees it, so this covers
/// that path and the union's own gather.
#[test]
fn take_from_empty_union_is_all_null() -> VortexResult<()> {
    let variants = variants()?;
    let nullability = Nullability::Nullable;
    let empty = UnionArray::empty(variants.clone(), nullability).into_array();
    let indices = PrimitiveArray::from_option_iter([None::<u64>, None]).into_array();

    let via_take = empty
        .take(indices.clone())?
        .execute::<Canonical>(&mut array_session().create_execution_ctx())?
        .into_union();
    let via_reduce = <Union as TakeReduce>::take(empty.as_::<Union>(), &indices)?
        .ok_or_else(|| vortex_err!("Union take must never decline"))?
        .as_::<Union>()
        .into_owned();

    let expected = || vec![Scalar::null(DType::Union(variants.clone(), nullability)); 2];

    assert_rows(&via_take, expected())?;
    assert_rows(&via_reduce, expected())
}

#[rstest]
#[case::non_nullable(union_array())]
#[case::nullable(nullable_union_array())]
fn take_conformance(#[case] array: VortexResult<UnionArray>) -> VortexResult<()> {
    test_take_conformance(
        &array?.into_array(),
        &mut array_session().create_execution_ctx(),
    );

    Ok(())
}

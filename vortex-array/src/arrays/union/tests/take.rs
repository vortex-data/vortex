// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take coverage for [`UnionArray`]: variant selection, outer nulls, and the empty source.

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

/// Take `indices` from `array`, asserting that the result reduced back to a union rather than
/// staying a lazy dictionary.
#[track_caller]
fn take(array: &UnionArray, indices: ArrayRef) -> VortexResult<UnionArray> {
    Ok(array
        .clone()
        .into_array()
        .take(indices)?
        .as_::<Union>()
        .into_owned())
}

#[test]
fn take_reorders_and_repeats_variant_selection() -> VortexResult<()> {
    let taken = take(
        &union_array()?,
        PrimitiveArray::from_iter([2u64, 1, 0, 1]).into_array(),
    )?;
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(
        taken.dtype(),
        &DType::Union(variants()?, Nullability::NonNullable)
    );
    assert_eq!(
        taken.execute_scalar(0, &mut ctx)?,
        Scalar::union(variants()?, 5, 30i32.into(), Nullability::NonNullable)?
    );
    assert_eq!(
        taken.execute_scalar(1, &mut ctx)?,
        Scalar::union(variants()?, 9, true.into(), Nullability::NonNullable)?
    );
    assert_eq!(
        taken.execute_scalar(2, &mut ctx)?,
        Scalar::union(variants()?, 5, 10i32.into(), Nullability::NonNullable)?
    );
    assert_eq!(
        taken.execute_scalar(3, &mut ctx)?,
        Scalar::union(variants()?, 9, true.into(), Nullability::NonNullable)?
    );

    Ok(())
}

#[test]
fn null_indices_become_outer_nulls_and_leave_children_alone() -> VortexResult<()> {
    let taken = take(
        &union_array()?,
        PrimitiveArray::from_option_iter([Some(1u64), None, Some(0)]).into_array(),
    )?;
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(
        taken.dtype(),
        &DType::Union(variants()?, Nullability::Nullable)
    );

    // A nullable index widens the union, but never the variants: the type IDs own the outer
    // nullability and the sparse children keep the dtypes the schema declares.
    assert_eq!(
        taken.child_by_name("number")?.dtype(),
        &DType::Primitive(PType::I32, Nullability::NonNullable)
    );
    assert_eq!(
        taken.child_by_name("flag")?.dtype(),
        &DType::Bool(Nullability::NonNullable)
    );

    assert_eq!(
        taken.execute_scalar(0, &mut ctx)?,
        Scalar::union(variants()?, 9, true.into(), Nullability::Nullable)?
    );
    assert_eq!(
        taken.execute_scalar(1, &mut ctx)?,
        Scalar::null(DType::Union(variants()?, Nullability::Nullable))
    );
    assert_eq!(
        taken.execute_scalar(2, &mut ctx)?,
        Scalar::union(variants()?, 5, 10i32.into(), Nullability::Nullable)?
    );

    Ok(())
}

#[test]
fn take_keeps_outer_and_inner_nulls_distinct() -> VortexResult<()> {
    let variants = nullable_variants()?;

    // Row 1 is an outer null and row 2 is a present union selecting a null child.
    let taken = take(
        &nullable_union_array()?,
        PrimitiveArray::from_iter([1u64, 2, 3]).into_array(),
    )?;
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(
        taken.execute_scalar(0, &mut ctx)?,
        Scalar::null(DType::Union(variants.clone(), Nullability::Nullable))
    );
    assert_eq!(
        taken.execute_scalar(1, &mut ctx)?,
        Scalar::union(
            variants.clone(),
            9,
            Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
            Nullability::Nullable,
        )?
    );
    assert_eq!(
        taken.execute_scalar(2, &mut ctx)?,
        Scalar::union(
            variants,
            9,
            Scalar::primitive(40i64, Nullability::Nullable),
            Nullability::Nullable,
        )?
    );

    Ok(())
}

/// An empty union has no row for its sparse children to point at, so an all-null gather must
/// synthesize placeholders instead. `take` short-circuits an empty source into a constant before
/// the union ever sees it, so cover that path and the union's own gather together.
#[test]
fn take_from_empty_union_is_all_null() -> VortexResult<()> {
    let empty = UnionArray::empty(variants()?, Nullability::Nullable).into_array();
    let indices = PrimitiveArray::from_option_iter([None::<u64>, None]).into_array();
    let mut ctx = array_session().create_execution_ctx();

    let via_take = empty
        .take(indices.clone())?
        .execute::<Canonical>(&mut ctx)?
        .into_union();
    let via_reduce = <Union as TakeReduce>::take(empty.as_::<Union>(), &indices)?
        .ok_or_else(|| vortex_err!("Union take must never decline"))?
        .as_::<Union>()
        .into_owned();

    for taken in [via_take, via_reduce] {
        assert_eq!(taken.len(), 2);
        assert_eq!(
            taken.dtype(),
            &DType::Union(variants()?, Nullability::Nullable)
        );

        for index in 0..taken.len() {
            assert!(taken.execute_scalar(index, &mut ctx)?.is_null());
        }
    }

    Ok(())
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

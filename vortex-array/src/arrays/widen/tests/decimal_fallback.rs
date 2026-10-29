// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conformance and canonical fallbacks for decimal widening.
//!
//! Arithmetic, zip, chunked and interleave execution preserve the logical decimal dtype and nulls.

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::SESSION;
use super::canonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::DecimalArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::compute::conformance::binary_numeric::test_binary_numeric_array;
use crate::compute::conformance::cast::test_cast_conformance;
use crate::compute::conformance::consistency::test_array_consistency;
use crate::compute::conformance::filter::test_filter_conformance;
use crate::compute::conformance::mask::test_mask_conformance;
use crate::compute::conformance::take::test_take_conformance;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

fn decimal_widen(validity: Validity) -> VortexResult<ArrayRef> {
    let values = DecimalArray::new(
        buffer![-125i16, 50, 50, 250, 900],
        DecimalDType::new(4, 2),
        validity,
    );
    let dtype = DType::Decimal(DecimalDType::new(18, 2), values.dtype().nullability());

    Ok(WidenArray::try_new(values.into_array(), dtype)?.into_array())
}

#[rstest]
#[case::non_nullable(Validity::NonNullable)]
#[case::partially_valid(Validity::from_iter([true, false, true, true, false]))]
fn test_decimal_conformance(#[case] validity: Validity) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = decimal_widen(validity)?;
    assert!(array.is::<Widen>());

    test_array_consistency(&array, &mut ctx);
    test_cast_conformance(&array, &mut ctx);
    test_filter_conformance(&array, &mut ctx);
    test_take_conformance(&array, &mut ctx);
    test_mask_conformance(&array, &mut ctx);
    test_binary_numeric_array(&array, &mut ctx);

    Ok(())
}

#[rstest]
#[case::non_nullable(Validity::NonNullable)]
#[case::partially_valid(Validity::from_iter([true, false, true, true, false]))]
fn test_decimal_arithmetic_falls_back_to_canonical(
    #[case] validity: Validity,
    #[values(Operator::Add, Operator::Sub, Operator::Mul, Operator::Div)] operator: Operator,
    #[values(false, true)] widen_on_right: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = decimal_widen(validity)?;
    let other = DecimalArray::from_iter([25i64, 50, -100, 125, -200], DecimalDType::new(18, 2))
        .into_array();
    let (lhs, rhs) = if widen_on_right {
        (other, array)
    } else {
        (array, other)
    };
    let expected = canonical(&lhs, &mut ctx)?
        .binary(canonical(&rhs, &mut ctx)?, operator)?
        .execute::<Canonical>(&mut ctx)?;

    let actual = lhs.binary(rhs, operator)?.execute::<Canonical>(&mut ctx)?;

    assert_eq!(actual.dtype(), expected.dtype());
    assert_arrays_eq!(actual, expected, &mut ctx);

    Ok(())
}

#[rstest]
#[case::true_branch(true, false)]
#[case::false_branch(false, true)]
#[case::both_branches(true, true)]
fn test_decimal_zip_falls_back_to_canonical(
    #[case] widen_if_true: bool,
    #[case] widen_if_false: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let if_true = decimal_widen(Validity::from_iter([true, false, true, true, false]))?;
    let if_false = WidenArray::try_new(
        DecimalArray::from_iter([500i16, -600, 700, -800, 900], DecimalDType::new(4, 2))
            .into_array(),
        DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable),
    )?
    .into_array();
    let canonical_true = canonical(&if_true, &mut ctx)?;
    let canonical_false = canonical(&if_false, &mut ctx)?;
    let mask =
        BoolArray::from_iter([Some(true), Some(true), None, Some(false), Some(true)]).into_array();
    let expected = mask
        .zip(canonical_true.clone(), canonical_false.clone())?
        .execute::<Canonical>(&mut ctx)?;
    let if_true = if widen_if_true {
        if_true
    } else {
        canonical_true
    };
    let if_false = if widen_if_false {
        if_false
    } else {
        canonical_false
    };

    let actual = mask
        .zip(if_true, if_false)?
        .execute::<Canonical>(&mut ctx)?;

    assert_eq!(actual.dtype(), expected.dtype());
    assert_arrays_eq!(actual, expected, &mut ctx);

    Ok(())
}

#[rstest]
#[case::non_nullable(Validity::NonNullable)]
#[case::partially_valid(Validity::from_iter([true, false, true, true, false]))]
fn test_decimal_chunked_execution_matches_canonical(
    #[case] validity: Validity,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = decimal_widen(validity)?;
    let expected = canonical(&array, &mut ctx)?;
    let chunks = ChunkedArray::from_iter([
        array.slice(0..2)?,
        expected.slice(2..3)?,
        array.slice(3..5)?,
    ])
    .into_array();

    let actual = chunks.execute::<DecimalArray>(&mut ctx)?;

    assert_eq!(actual.dtype(), expected.dtype());
    assert_arrays_eq!(actual, expected, &mut ctx);

    Ok(())
}

#[rstest]
#[case::non_nullable(Validity::NonNullable)]
#[case::partially_valid(Validity::from_iter([true, false, true, true, false]))]
fn test_decimal_interleave_falls_back_to_canonical(#[case] validity: Validity) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = decimal_widen(validity)?;
    let other = DecimalArray::from_iter([500i64, -600, 700, -800, 950], DecimalDType::new(18, 2))
        .into_array();
    let array_indices = buffer![0u8, 1, 0, 1, 0, 0, 1].into_array();
    let row_indices = buffer![4u32, 0, 1, 4, 0, 4, 1].into_array();
    let expected = array_indices
        .interleave(
            [canonical(&array, &mut ctx)?, other.clone()],
            row_indices.clone(),
        )?
        .execute::<DecimalArray>(&mut ctx)?;

    let actual = array_indices
        .interleave([array, other], row_indices)?
        .execute::<DecimalArray>(&mut ctx)?;

    assert_eq!(actual.dtype(), expected.dtype());
    assert_arrays_eq!(actual, expected, &mut ctx);

    Ok(())
}

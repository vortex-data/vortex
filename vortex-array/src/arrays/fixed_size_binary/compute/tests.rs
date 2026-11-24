// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for the compute kernels and aggregates over fixed-size binary arrays.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_constant::is_constant;
use crate::aggregate_fn::fns::is_sorted::is_sorted;
use crate::aggregate_fn::fns::is_sorted::is_strict_sorted;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::FixedSizeBinaryArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::scalar_fn::builtin::operators::CompareOperator;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// Values of width `W` whose lexicographic order differs from their order as little-endian
/// integers, so a comparison that reinterprets the bytes as numbers would fail.
fn values<const W: usize>() -> [[u8; W]; 4] {
    let mut values = [[0u8; W]; 4];
    values[0][0] = 1;
    values[1][W - 1] = 2;
    values[2][0] = 1;
    values[2][W - 1] = 1;
    values[3] = [0xff; W];
    values
}

fn compare_case<const W: usize>(op: CompareOperator) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let width = u32::try_from(W).vortex_expect("test width fits in u32");
    let lhs = values::<W>();
    let mut rhs = lhs;
    rhs.rotate_left(1);

    let result = FixedSizeBinaryArray::from_iter(lhs, width)
        .into_array()
        .compare(FixedSizeBinaryArray::from_iter(rhs, width).into_array(), op)?;
    let expected = BoolArray::from_iter(
        lhs.iter()
            .zip(rhs.iter())
            .map(|(l, r)| op_holds(op, l.as_slice(), r.as_slice())),
    );
    assert_arrays_eq!(result, expected, &mut ctx);

    let constant = Scalar::fixed_size_binary(lhs[2].to_vec(), Nullability::NonNullable);
    let result = FixedSizeBinaryArray::from_iter(lhs, width)
        .into_array()
        .compare(ConstantArray::new(constant, lhs.len()).into_array(), op)?;
    let expected = BoolArray::from_iter(
        lhs.iter()
            .map(|l| op_holds(op, l.as_slice(), lhs[2].as_slice())),
    );
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

fn op_holds(op: CompareOperator, lhs: &[u8], rhs: &[u8]) -> bool {
    let ordering = lhs.cmp(rhs);
    match op {
        CompareOperator::Eq => ordering.is_eq(),
        CompareOperator::NotEq => ordering.is_ne(),
        CompareOperator::Gt => ordering.is_gt(),
        CompareOperator::Gte => ordering.is_ge(),
        CompareOperator::Lt => ordering.is_lt(),
        CompareOperator::Lte => ordering.is_le(),
    }
}

#[rstest]
fn compare_power_of_two_width(
    #[values(
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Gt,
        CompareOperator::Gte,
        CompareOperator::Lt,
        CompareOperator::Lte
    )]
    op: CompareOperator,
) -> VortexResult<()> {
    compare_case::<4>(op)?;
    compare_case::<16>(op)
}

#[rstest]
fn compare_other_width(
    #[values(
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Gt,
        CompareOperator::Gte,
        CompareOperator::Lt,
        CompareOperator::Lte
    )]
    op: CompareOperator,
) -> VortexResult<()> {
    compare_case::<3>(op)?;
    compare_case::<20>(op)
}

#[test]
fn compare_propagates_nulls() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = FixedSizeBinaryArray::from_option_iter([Some(b"ab"), None, Some(b"cd")], 2);
    let rhs = FixedSizeBinaryArray::from_option_iter([Some(b"ab"), Some(b"ab"), None], 2);
    let result = lhs
        .into_array()
        .compare(rhs.into_array(), CompareOperator::Eq)?;
    let expected = BoolArray::from_iter([Some(true), None, None]);
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn min_max_by_byte() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = FixedSizeBinaryArray::from_option_iter(
        [Some(b"b\x00"), None, Some(b"a\xff"), Some(b"b\x01")],
        2,
    )
    .into_array();
    let result = min_max(&array, &mut ctx, NumericalAggregateOpts::default())?
        .expect("array has valid values");
    assert_eq!(
        result.min,
        Scalar::fixed_size_binary(b"a\xff".to_vec(), Nullability::NonNullable)
    );
    assert_eq!(
        result.max,
        Scalar::fixed_size_binary(b"b\x01".to_vec(), Nullability::NonNullable)
    );

    let all_null = FixedSizeBinaryArray::from_option_iter([None::<&[u8]>, None], 2).into_array();
    assert!(min_max(&all_null, &mut ctx, NumericalAggregateOpts::default())?.is_none());
    Ok(())
}

#[rstest]
#[case::sorted(FixedSizeBinaryArray::from_iter([b"aa", b"ab", b"ab", b"ba"], 2).into_array(), true, false)]
#[case::strict(FixedSizeBinaryArray::from_iter([b"aa", b"ab", b"ba"], 2).into_array(), true, true)]
#[case::unsorted(FixedSizeBinaryArray::from_iter([b"ab", b"aa"], 2).into_array(), false, false)]
#[case::nulls_first(
    FixedSizeBinaryArray::from_option_iter([None, Some(b"aa"), Some(b"ab")], 2).into_array(),
    true,
    true,
)]
#[case::null_after_value(
    FixedSizeBinaryArray::from_option_iter([Some(b"aa"), None], 2).into_array(),
    false,
    false,
)]
fn sortedness(
    #[case] array: ArrayRef,
    #[case] sorted: bool,
    #[case] strict: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(is_sorted(&array, &mut ctx)?, sorted);
    assert_eq!(is_strict_sorted(&array, &mut ctx)?, strict);
    Ok(())
}

#[rstest]
#[case::constant(FixedSizeBinaryArray::from_iter([b"abc"; 4], 3).into_array(), true)]
#[case::varying(FixedSizeBinaryArray::from_iter([b"abc", b"abd"], 3).into_array(), false)]
#[case::mixed_nulls(
    FixedSizeBinaryArray::from_option_iter([Some(b"abc"), None], 3).into_array(),
    false,
)]
#[case::zero_width(FixedSizeBinaryArray::from_iter([b""; 3], 0).into_array(), true)]
fn constancy(#[case] array: ArrayRef, #[case] expected: bool) -> VortexResult<()> {
    assert_eq!(
        is_constant(&array, &mut SESSION.create_execution_ctx())?,
        expected
    );
    Ok(())
}

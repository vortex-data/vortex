// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::*;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

fn accumulator(dtype: DType) -> VortexResult<Accumulator<ApproxDistinctCount>> {
    Accumulator::try_new(ApproxDistinctCount, EmptyOptions, dtype)
}

fn result(acc: &mut Accumulator<ApproxDistinctCount>) -> VortexResult<Option<u64>> {
    Ok(acc.finish()?.as_primitive().typed_value::<u64>())
}

#[rstest]
#[case::primitive(
    PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(1), None, Some(3)]).into_array(),
    2
)]
#[case::all_null(PrimitiveArray::from_option_iter([None::<i64>, None]).into_array(), 0)]
#[case::floats(PrimitiveArray::new(buffer![1.0f64, 2.0, 1.0, f64::NAN], Validity::NonNullable).into_array(), 3)]
#[case::bool(BoolArray::from_iter([Some(true), None, Some(true)]).into_array(), 1)]
#[case::utf8(
    VarBinViewArray::from_iter_nullable_str([
        Some("a"),
        Some("a string longer than twelve bytes"),
        None,
        Some("a"),
        Some("a string longer than twelve bytes"),
    ])
    .into_array(),
    2
)]
#[case::decimal(
    DecimalArray::new(buffer![10i64, 20, 10, -1], DecimalDType::new(10, 2), Validity::NonNullable).into_array(),
    3
)]
#[case::constant(ConstantArray::new(7u8, 10).into_array(), 1)]
#[case::null_constant(
    ConstantArray::new(Scalar::null(DType::Primitive(PType::U8, Nullability::Nullable)), 10).into_array(),
    0
)]
fn counts_distinct_non_null_values(
    #[case] array: ArrayRef,
    #[case] expected: u64,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(approx_distinct_count(&array, &mut ctx)?, expected);
    Ok(())
}

#[test]
fn unsupported_dtype() {
    assert!(!ApproxDistinctCount::supports_dtype(&DType::Null));
    assert!(!ApproxDistinctCount::supports_dtype(&DType::Decimal(
        DecimalDType::new(40, 0),
        NonNullable
    )));
}

/// Batches merge exactly, so values repeated across batches are counted once.
#[test]
fn exact_across_batches() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut acc = accumulator(DType::Primitive(PType::U32, NonNullable))?;
    acc.accumulate(&buffer![1u32, 2, 3].into_array(), &mut ctx)?;
    acc.accumulate(&buffer![2u32, 3, 4].into_array(), &mut ctx)?;
    assert_eq!(result(&mut acc)?, Some(4));
    Ok(())
}

/// Two exact accumulators merge by union.
#[test]
fn merge_exact_unions() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DType::Primitive(PType::U32, NonNullable);
    let mut a = accumulator(dtype.clone())?;
    let mut b = accumulator(dtype)?;
    a.accumulate(&buffer![1u32, 2, 3].into_array(), &mut ctx)?;
    b.accumulate(&buffer![2u32, 3, 4].into_array(), &mut ctx)?;
    a.merge_from(&mut b)?;
    assert_eq!(result(&mut a)?, Some(4));
    Ok(())
}

/// The serialized partial holds only the count, so merging it is inexact and adds counts.
#[test]
fn merge_serialized_partial_adds_counts() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DType::Primitive(PType::U32, NonNullable);
    let mut a = accumulator(dtype.clone())?;
    a.accumulate(&buffer![1u32, 2, 3].into_array(), &mut ctx)?;
    let partial = a.flush()?;
    assert_eq!(partial.as_primitive().typed_value::<u64>(), Some(3));

    let mut b = accumulator(dtype)?;
    b.accumulate(&buffer![2u32, 3, 4].into_array(), &mut ctx)?;
    b.combine_partials(&partial)?;
    assert_eq!(result(&mut b)?, Some(6));
    Ok(())
}

/// Once inexact, each batch adds its own distinct count.
#[test]
fn inexact_adds_batch_counts() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut acc = accumulator(DType::Utf8(NonNullable))?;
    acc.combine_partials(&Scalar::primitive(5u64, NonNullable))?;
    acc.accumulate(
        &VarBinViewArray::from_iter_str(["x", "y", "x"]).into_array(),
        &mut ctx,
    )?;
    assert_eq!(result(&mut acc)?, Some(7));
    Ok(())
}

/// An exact set that outgrows the bound keeps only its count.
#[test]
fn degrades_past_bound() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let n = MAX_EXACT_DISTINCT_VALUES as u64 + 1;
    let mut acc = accumulator(DType::Primitive(PType::U64, NonNullable))?;
    acc.accumulate(&Buffer::from_iter(0..n).into_array(), &mut ctx)?;
    // The repeated values are counted again, since the state no longer holds the set.
    acc.accumulate(&Buffer::from_iter(0..10u64).into_array(), &mut ctx)?;
    assert_eq!(result(&mut acc)?, Some(n + 10));
    Ok(())
}

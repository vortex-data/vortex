// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block fusion of DateTimeParts: its children combined block by block where it stands.

use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::assert_arrays_eq;
use vortex_array::block::fused_chain_count;
use vortex_array::dtype::DType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::test_harness::trace::trace_op;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_session::VortexSession;

use crate::DateTimeParts;
use crate::array::DateTimePartsArraySlotsExt;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_fastlanes::initialize(&session);
    crate::initialize(&session);
    session
});

const LEN: usize = 3000;

/// Microsecond timestamps over many days, with optional nulls and time zone.
fn timestamps(nullable: bool, tz: Option<&str>) -> TemporalArray {
    let values = (0..LEN as i64).map(|i| i * 7_919_000_123 + (i % 97) * 1_000);
    let values = if nullable {
        PrimitiveArray::from_option_iter(
            values.enumerate().map(|(i, v)| (i % 11 != 0).then_some(v)),
        )
    } else {
        PrimitiveArray::from_iter(values)
    };
    TemporalArray::new_timestamp(
        values.into_array(),
        TimeUnit::Microseconds,
        tz.map(Arc::from),
    )
}

/// DateTimeParts of `timestamps`, with its days child bit-packed under FoR when `encode_days`.
fn date_time_parts(nullable: bool, encode_days: bool, tz: Option<&str>) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = DateTimeParts::try_from_temporal(timestamps(nullable, tz), &mut ctx)?;
    if !encode_days {
        return Ok(parts.into_array());
    }
    let days = parts.days().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let (shifted, min) = match_each_integer_ptype!(days.ptype(), |T| {
        let min = days
            .as_slice::<T>()
            .iter()
            .copied()
            .min()
            .unwrap_or_default();
        let shifted = days
            .as_slice::<T>()
            .iter()
            .map(|&d| d - min)
            .collect::<Buffer<T>>();
        (
            PrimitiveArray::new(shifted, days.validity()?),
            Scalar::from(min),
        )
    });
    let bp = bitpack_encode(&shifted, 12, None, &mut ctx)?.into_array();
    let days = FoR::try_new(bp, min)?.into_array();
    Ok(DateTimeParts::try_new(
        parts.dtype().clone(),
        days,
        parts.seconds().clone(),
        parts.subseconds().clone(),
    )?
    .into_array())
}

/// Execute `array` with fusion on and off, check they agree, and return the slot executions and
/// fusion steps of the fused run's trace.
fn fused(array: ArrayRef) -> VortexResult<String> {
    let expected = array
        .clone()
        .execute::<Canonical>(&mut SESSION.create_execution_ctx().with_block_fusion(false))?
        .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let traced = trace_op(|| array.execute::<Canonical>(&mut ctx))?;
    let output = traced.output.into_array();
    assert_eq!(output.dtype(), expected.dtype());
    assert_arrays_eq!(output, expected, &mut ctx);
    Ok(traced
        .trace
        .to_string()
        .lines()
        .map(str::trim)
        .filter(|line| {
            ["ExecuteSlot", "ExecuteSpine", "fuse ", "drive ", "block_op"]
                .iter()
                .any(|p| line.starts_with(p))
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

/// The trace of a DateTimeParts array of `len` rows combined where it stands, with no child
/// executed first.
fn combined(dtype: &DType, len: usize) -> String {
    format!(
        "fuse start input=zip array=vortex.datetimeparts({dtype}, len={len})
drive input=zip ops=0 len={len}
block_op 0 copy i64->i64 scratch0 -> output"
    )
}

/// With FoR over BitPacked days, the children are combined where DateTimeParts stands, with no
/// parent above it: days unpacked block by block, seconds and subseconds lent from their
/// buffers, and the combined block copied to the output. No child executes on its own.
#[rstest]
#[case::plain(false, None)]
#[case::nullable(true, None)]
#[case::time_zone(false, Some("Europe/London"))]
fn children_combine_where_date_time_parts_stands(
    #[case] nullable: bool,
    #[case] tz: Option<&str>,
) -> VortexResult<()> {
    let array = date_time_parts(nullable, true, tz)?;
    let dtype = array.dtype().clone();
    assert_eq!(fused(array)?, combined(&dtype, LEN));
    Ok(())
}

/// A slice reaches the children, so the days source starts part way into a FastLanes chunk.
#[rstest]
#[case::mid_chunk(517..2900)]
#[case::within_a_block(1500..1600)]
#[case::to_the_end(1023..LEN)]
fn sliced_children_combine(#[case] rows: Range<usize>) -> VortexResult<()> {
    let array = date_time_parts(true, true, None)?.slice(rows.clone())?;
    let dtype = array.dtype().clone();
    assert_eq!(fused(array)?, combined(&dtype, rows.len()));
    Ok(())
}

/// With every child canonical, combining saves no buffer, so DateTimeParts decodes as before.
#[test]
fn canonical_children_decode_level_wise() -> VortexResult<()> {
    let array = date_time_parts(false, false, None)?;
    let before = fused_chain_count();
    assert_eq!(fused(array)?, "");
    assert_eq!(fused_chain_count(), before);
    Ok(())
}

/// Constant seconds and subseconds are folded into the combine, so days is its only input: a
/// constant child is neither canonical nor a block source, so it could not be one.
#[test]
fn constant_parts_fold_into_the_combine() -> VortexResult<()> {
    let parts = date_time_parts(false, true, None)?;
    let parts = parts.as_::<DateTimeParts>();
    let array = DateTimeParts::try_new(
        parts.dtype().clone(),
        parts.days().clone(),
        ConstantArray::new(Scalar::from(17i32), LEN).into_array(),
        ConstantArray::new(Scalar::from(5i32), LEN).into_array(),
    )?
    .into_array();
    let dtype = array.dtype().clone();
    assert_eq!(fused(array)?, combined(&dtype, LEN));
    Ok(())
}

/// Seconds that are neither canonical nor a block source execute first, and days, asked for
/// last, is then combined from its source rather than decoded on its own.
#[test]
fn time_parts_execute_before_days_combine() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = date_time_parts(false, true, None)?;
    let parts = parts.as_::<DateTimeParts>();
    let seconds = parts
        .seconds()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let codes = PrimitiveArray::from_iter((0u32..).take(LEN));
    let seconds = DictArray::try_new(codes.into_array(), seconds.into_array())?.into_array();
    let array = DateTimeParts::try_new(
        parts.dtype().clone(),
        parts.days().clone(),
        seconds,
        parts.subseconds().clone(),
    )?
    .into_array();
    let dtype = array.dtype().clone();
    assert_eq!(
        fused(array)?,
        format!(
            "ExecuteSlot slot=1 parent=vortex.datetimeparts({dtype}, len={LEN}) \
             child=vortex.dict(i32, len={LEN})\n{}",
            combined(&dtype, LEN)
        )
    );
    Ok(())
}

/// Seconds stored wider than `i32` are cast by the level-wise decode, so the parts do not
/// combine.
#[test]
fn wide_seconds_do_not_combine() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = date_time_parts(false, true, None)?;
    let parts = parts.as_::<DateTimeParts>();
    let seconds = parts
        .seconds()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let wide = PrimitiveArray::from_iter(seconds.as_slice::<i32>().iter().map(|&s| i64::from(s)));
    let array = DateTimeParts::try_new(
        parts.dtype().clone(),
        parts.days().clone(),
        wide.into_array(),
        parts.subseconds().clone(),
    )?
    .into_array();
    let before = fused_chain_count();
    assert!(!fused(array)?.contains("fuse"));
    assert_eq!(fused_chain_count(), before);
    Ok(())
}

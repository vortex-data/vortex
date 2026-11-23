// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block fusion of DateTimeParts: its children combined block by block where it stands.

use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::dtype::DType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::test_harness::fusion::fusion_trace;
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

/// Execute `array` with fusion off and on, check they agree, and return the fused run's
/// fusion steps.
fn fused(array: ArrayRef) -> VortexResult<String> {
    fusion_trace(array, &SESSION)
}

/// The trace of `array`, a DateTimeParts array over FoR over BitPacked days, fused where it
/// stands: FoR's unpack is the leaf, DateTimeParts' op combines each block of days with the same
/// rows of its time parts, and the chain is driven at the root.
fn combined(array: &ArrayRef) -> String {
    let days = array.as_::<DateTimeParts>().days().dtype();
    combined_rows(array.dtype(), days, array.len())
}

/// [`combined`] for `len` rows of a DateTimeParts array of `dtype` over days of `days`.
fn combined_rows(dtype: &DType, days: &DType, len: usize) -> String {
    let days_ptype = days.as_ptype();
    format!(
        "ExecuteSpine slot=0 parent=vortex.datetimeparts({dtype}, len={len}) \
         child=fastlanes.for.v2({days}, len={len})
fuse start array=fastlanes.for.v2({days}, len={len})
fuse extend parent=vortex.datetimeparts({dtype}, len={len}) ops=1
drive ops=1 len={len}
block_op 0 copy {days_ptype}->i64 scratch0 -> output"
    )
}

/// DateTimeParts asks for days as its spine once its time parts are canonical. FoR over BitPacked
/// days is the leaf, and DateTimeParts' op combines each block of days with the same rows of its
/// time parts, with no parent above it. Days never executes on its own.
#[rstest]
#[case::plain(false, None)]
#[case::nullable(true, None)]
#[case::time_zone(false, Some("Europe/London"))]
fn days_fuse_into_date_time_parts(
    #[case] nullable: bool,
    #[case] tz: Option<&str>,
) -> VortexResult<()> {
    let array = date_time_parts(nullable, true, tz)?;
    let expected = combined(&array);
    assert_eq!(fused(array)?, expected);
    Ok(())
}

/// A slice reaches the children, so the days source starts part way into a FastLanes chunk.
#[rstest]
#[case::mid_chunk(517..2900)]
#[case::within_a_block(1500..1600)]
#[case::to_the_end(1023..LEN)]
fn sliced_days_fuse(#[case] rows: Range<usize>) -> VortexResult<()> {
    let whole = date_time_parts(true, true, None)?;
    let days = whole.as_::<DateTimeParts>().days().dtype().clone();
    let expected = combined_rows(whole.dtype(), &days, rows.len());
    let array = whole.slice(rows)?;
    assert_eq!(fused(array)?, expected);
    Ok(())
}

/// With days already canonical there is nothing to fuse, so DateTimeParts decodes as before.
#[test]
fn canonical_children_decode_level_wise() -> VortexResult<()> {
    let array = date_time_parts(false, false, None)?;
    assert_eq!(fused(array)?, "");
    Ok(())
}

/// Constant time parts execute into buffers first, as level-wise decoding does, and days then
/// fuses as usual.
#[test]
fn constant_time_parts_execute_first() -> VortexResult<()> {
    let parts = date_time_parts(false, true, None)?;
    let parts = parts.as_::<DateTimeParts>();
    let array = DateTimeParts::try_new(
        parts.dtype().clone(),
        parts.days().clone(),
        ConstantArray::new(Scalar::from(17i32), LEN).into_array(),
        ConstantArray::new(Scalar::from(5i32), LEN).into_array(),
    )?
    .into_array();
    let (dtype, expected) = (array.dtype().clone(), combined(&array));
    assert_eq!(
        fused(array)?,
        format!(
            "ExecuteSlot slot=1 parent=vortex.datetimeparts({dtype}, len={LEN}) \
             child=vortex.constant(i32, len={LEN})
ExecuteSlot slot=2 parent=vortex.datetimeparts({dtype}, len={LEN}) \
             child=vortex.constant(i32, len={LEN})
{expected}"
        )
    );
    Ok(())
}

/// Seconds that are not canonical execute first, and days, asked for last, then fuses from its
/// source rather than being decoded on its own.
#[test]
fn time_parts_execute_before_days_fuses() -> VortexResult<()> {
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
    let (dtype, expected) = (array.dtype().clone(), combined(&array));
    assert_eq!(
        fused(array)?,
        format!(
            "ExecuteSlot slot=1 parent=vortex.datetimeparts({dtype}, len={LEN}) \
             child=vortex.dict(i32, len={LEN})\n{expected}"
        )
    );
    Ok(())
}

/// Seconds stored wider than `i32` are cast by the level-wise decode, so DateTimeParts has no op
/// for its days, and nothing fuses.
#[test]
fn wide_seconds_do_not_fuse() -> VortexResult<()> {
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
    let (dtype, days) = (array.dtype().clone(), parts.days().dtype().clone());
    // DateTimeParts is asked for its op before days builds a source, so days builds none.
    assert_eq!(
        fused(array)?,
        format!(
            "ExecuteSpine slot=0 parent=vortex.datetimeparts({dtype}, len={LEN}) \
             child=fastlanes.for.v2({days}, len={LEN})"
        )
    );
    Ok(())
}

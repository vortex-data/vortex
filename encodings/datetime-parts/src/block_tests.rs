// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block fusion of DateTimeParts: its children combined block by block where it stands.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::assert_arrays_eq;
use vortex_array::block::fused_chain_count;
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

/// Microsecond timestamps over many days, with optional nulls.
fn timestamps(nullable: bool) -> TemporalArray {
    let values = (0..LEN as i64).map(|i| i * 7_919_000_123 + (i % 97) * 1_000);
    let values = if nullable {
        PrimitiveArray::from_option_iter(
            values.enumerate().map(|(i, v)| (i % 11 != 0).then_some(v)),
        )
    } else {
        PrimitiveArray::from_iter(values)
    };
    TemporalArray::new_timestamp(values.into_array(), TimeUnit::Microseconds, None)
}

/// DateTimeParts of `timestamps`, with its days child bit-packed under FoR when `encode_days`.
fn date_time_parts(nullable: bool, encode_days: bool) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = DateTimeParts::try_from_temporal(timestamps(nullable), &mut ctx)?;
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

/// Execute `array` with fusion on and off, check they agree, and return the fusion steps of the
/// fused run's trace.
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
            ["fuse ", "drive ", "block_op"]
                .iter()
                .any(|p| line.starts_with(p))
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

/// The children are combined where DateTimeParts stands, with no parent above it: days from
/// FoR's fused unpack of BitPacked or from its buffer, seconds and subseconds from theirs, and
/// the combined block copied to the output.
#[rstest]
#[case::canonical_days(false, false)]
#[case::encoded_days(false, true)]
#[case::nullable(true, true)]
fn children_combine_where_date_time_parts_stands(
    #[case] nullable: bool,
    #[case] encode_days: bool,
) -> VortexResult<()> {
    let array = date_time_parts(nullable, encode_days)?;
    let dtype = array.dtype().clone();
    let trace = fused(array)?;
    assert_eq!(
        trace,
        format!(
            "fuse start input=zip array=vortex.datetimeparts({dtype}, len={LEN})
drive input=zip ops=0 len={LEN}
block_op 0 copy i64->i64 scratch0 -> output"
        )
    );
    Ok(())
}

/// Constant seconds and subseconds are folded into the combine, so days is its only input.
#[test]
fn constant_parts_fold_into_the_combine() -> VortexResult<()> {
    let parts = date_time_parts(false, true)?;
    let parts = parts.as_::<DateTimeParts>();
    let array = DateTimeParts::try_new(
        parts.dtype().clone(),
        parts.days().clone(),
        ConstantArray::new(Scalar::from(17i32), LEN).into_array(),
        ConstantArray::new(Scalar::from(5i32), LEN).into_array(),
    )?
    .into_array();
    assert!(fused(array)?.starts_with("fuse start input=zip"));
    Ok(())
}

/// Seconds stored wider than `i32` are cast by the level-wise decode, so the parts do not
/// combine.
#[test]
fn wide_seconds_do_not_combine() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = date_time_parts(false, false)?;
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
    assert_eq!(fused(array)?, "");
    assert_eq!(fused_chain_count(), before);
    Ok(())
}

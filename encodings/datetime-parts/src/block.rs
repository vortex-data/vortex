// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DateTimeParts' op for a [`BlockProgram`](vortex_array::block::BlockProgram): each block of
//! days combines with the same rows of the time parts into timestamps.

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::arrays::Primitive;
use vortex_array::block::Block;
use vortex_array::block::BlockOp;
use vortex_array::block::Values;
use vortex_array::block::Written;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_compute::lane_kernels::Repeat;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::canonical::combine_parts;

/// The op of a DateTimeParts array of `dtype` over its days child of `days_dtype`, once its
/// `seconds` and `subseconds` are executed: it combines each block of days with the same rows of
/// the time parts, as [`decode_to_temporal`](crate::canonical::decode_to_temporal) does.
///
/// `None` when it decodes some other way: a unit of days, or a time part that is neither a
/// non-null constant nor canonical `i32` values, which the level-wise decode casts.
pub(crate) fn days_block_op(
    dtype: &DType,
    days_dtype: &DType,
    seconds: &ArrayRef,
    subseconds: &ArrayRef,
) -> Option<Box<dyn BlockOp>> {
    let DType::Extension(ext) = dtype else {
        return None;
    };
    let divisor = match ext.metadata_opt::<Timestamp>()?.unit {
        TimeUnit::Nanoseconds => 1_000_000_000,
        TimeUnit::Microseconds => 1_000_000,
        TimeUnit::Milliseconds => 1_000,
        TimeUnit::Seconds => 1,
        TimeUnit::Days => return None,
    };
    let DType::Primitive(days_ptype, _) = days_dtype else {
        return None;
    };
    if !days_ptype.is_int() {
        return None;
    }
    let (seconds, subseconds) = (TimeColumn::new(seconds)?, TimeColumn::new(subseconds)?);
    Some(match_each_integer_ptype!(*days_ptype, |D| {
        combine_op::<D>(seconds, subseconds, divisor)
    }))
}

fn combine_op<D: NativePType + AsPrimitive<i64>>(
    seconds: TimeColumn,
    subseconds: TimeColumn,
    divisor: i64,
) -> Box<dyn BlockOp> {
    Box::new(CombineParts::<D> {
        seconds,
        subseconds,
        divisor,
        _days: PhantomData,
    })
}

/// A seconds or subseconds part of every row.
enum TimeColumn {
    /// One value shared by every row.
    Constant(i64),
    /// Canonical `i32` values.
    Values(Buffer<i32>),
}

impl TimeColumn {
    fn new(part: &ArrayRef) -> Option<Self> {
        if let Some(constant) = part.as_constant() {
            return constant.as_primitive().as_::<i64>().map(Self::Constant);
        }
        (part.dtype() == &DType::Primitive(PType::I32, Nullability::NonNullable))
            .then(|| {
                part.as_opt::<Primitive>()
                    .map(|part| part.to_buffer::<i32>())
            })
            .flatten()
            .map(Self::Values)
    }
}

/// Combines blocks of days with the same rows of seconds and subseconds.
struct CombineParts<D> {
    seconds: TimeColumn,
    subseconds: TimeColumn,
    divisor: i64,
    _days: PhantomData<fn(D)>,
}

impl<D: NativePType + AsPrimitive<i64>> BlockOp for CombineParts<D> {
    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>> {
        let start = block.start();
        let Values::Copy { input, out } = block.values else {
            vortex_bail!("DateTimeParts combines days into a block of its own");
        };
        let days = input.as_slice::<D>();
        let (len, divisor) = (days.len(), self.divisor);
        let rows = start..start + len;
        let combine = |values: &mut [MaybeUninit<i64>]| match (&self.seconds, &self.subseconds) {
            (TimeColumn::Constant(seconds), TimeColumn::Constant(subseconds)) => combine_parts(
                days,
                Repeat::new(*seconds, len),
                Repeat::new(*subseconds, len),
                divisor,
                values,
            ),
            (TimeColumn::Constant(seconds), TimeColumn::Values(subseconds)) => combine_parts(
                days,
                Repeat::new(*seconds, len),
                &subseconds[rows],
                divisor,
                values,
            ),
            (TimeColumn::Values(seconds), TimeColumn::Constant(subseconds)) => combine_parts(
                days,
                &seconds[rows],
                Repeat::new(*subseconds, len),
                divisor,
                values,
            ),
            (TimeColumn::Values(seconds), TimeColumn::Values(subseconds)) => combine_parts(
                days,
                &seconds[rows.clone()],
                &subseconds[rows],
                divisor,
                values,
            ),
        };
        // SAFETY: `combine_parts` writes every value it is handed.
        Ok(unsafe { out.fill_with(combine) })
    }

    /// The op reads days as one width and writes timestamps as another.
    fn supports_in_place(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
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

    /// DateTimeParts of microsecond timestamps over many days, with optional nulls, whose days are
    /// bit-packed under FoR.
    fn date_time_parts(nullable: bool, tz: Option<&str>) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0..LEN as i64).map(|i| i * 7_919_000_123 + (i % 97) * 1_000);
        let values = if nullable {
            PrimitiveArray::from_option_iter(
                values.enumerate().map(|(i, v)| (i % 11 != 0).then_some(v)),
            )
        } else {
            PrimitiveArray::from_iter(values)
        };
        let temporal = TemporalArray::new_timestamp(
            values.into_array(),
            TimeUnit::Microseconds,
            tz.map(Arc::from),
        );
        let parts = DateTimeParts::try_from_temporal(temporal, &mut ctx)?;
        let days = parts.days().clone().execute::<PrimitiveArray>(&mut ctx)?;
        let days = match_each_integer_ptype!(days.ptype(), |T| { for_over_bitpacked::<T>(&days)? });
        Ok(DateTimeParts::try_new(
            parts.dtype().clone(),
            days,
            parts.seconds().clone(),
            parts.subseconds().clone(),
        )?
        .into_array())
    }

    /// `values` shifted by their minimum and bit-packed at width 12, under FoR with that minimum.
    fn for_over_bitpacked<T>(values: &PrimitiveArray) -> VortexResult<ArrayRef>
    where
        T: vortex_array::dtype::NativePType + num_traits::PrimInt,
        Scalar: From<T>,
    {
        let mut ctx = SESSION.create_execution_ctx();
        let slice = values.as_slice::<T>();
        let min = slice.iter().copied().min().unwrap_or_else(T::zero);
        let shifted = PrimitiveArray::new(
            slice.iter().map(|&d| d - min).collect::<Buffer<T>>(),
            values.validity()?,
        );
        let bp = bitpack_encode(&shifted, 12, None, &mut ctx)?.into_array();
        Ok(FoR::try_new(bp, Scalar::from(min))?.into_array())
    }

    /// The executor's trace for `len` rows of a DateTimeParts array of `dtype` over FoR over
    /// BitPacked days of `days`, fused where it stands: FoR's unpack is the leaf, DateTimeParts'
    /// op combines each block of days with the same rows of its time parts, and the chain is
    /// driven at the root.
    fn combined(dtype: &DType, days: &DType, len: usize) -> String {
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

    /// [`combined`] for the whole of `array`.
    fn combined_array(array: &ArrayRef) -> String {
        let days = array.as_::<DateTimeParts>().days().dtype();
        combined(array.dtype(), days, array.len())
    }

    /// DateTimeParts asks for days as its spine once its time parts are
    /// columnar, and days never executes on its own.
    #[rstest]
    #[case::plain(false, None)]
    #[case::nullable(true, None)]
    #[case::time_zone(false, Some("Europe/London"))]
    fn days_fuse(#[case] nullable: bool, #[case] tz: Option<&str>) -> VortexResult<()> {
        let array = date_time_parts(nullable, tz)?;
        let expected = combined_array(&array);
        assert_eq!(fusion_trace(array, &SESSION)?, expected);
        Ok(())
    }

    /// A slice reaches the children, so the days source starts part way into a FastLanes chunk.
    #[rstest]
    #[case::mid_chunk(517..2900)]
    #[case::within_a_block(1500..1600)]
    #[case::to_the_end(1023..LEN)]
    fn sliced_days_fuse(#[case] rows: Range<usize>) -> VortexResult<()> {
        let whole = date_time_parts(true, None)?;
        let days = whole.as_::<DateTimeParts>().days().dtype().clone();
        let expected = combined(whole.dtype(), &days, rows.len());
        assert_eq!(fusion_trace(whole.slice(rows)?, &SESSION)?, expected);
        Ok(())
    }

    /// Constant time parts stay constant, so days fuses straight away.
    #[test]
    fn days_fuse_with_constant_time_parts() -> VortexResult<()> {
        let parts = date_time_parts(false, None)?;
        let parts = parts.as_::<DateTimeParts>();
        let array = DateTimeParts::try_new(
            parts.dtype().clone(),
            parts.days().clone(),
            ConstantArray::new(Scalar::from(17i32), LEN).into_array(),
            ConstantArray::new(Scalar::from(5i32), LEN).into_array(),
        )?
        .into_array();
        let expected = combined_array(&array);
        assert_eq!(fusion_trace(array, &SESSION)?, expected);
        Ok(())
    }

    /// Seconds that are not columnar execute first, and days, asked for last, then fuses from its
    /// source rather than being decoded on its own.
    #[test]
    fn time_parts_execute_before_days_fuse() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let parts = date_time_parts(false, None)?;
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
        let (dtype, expected) = (array.dtype().clone(), combined_array(&array));
        assert_eq!(
            fusion_trace(array, &SESSION)?,
            format!(
                "ExecuteSlot slot=1 parent=vortex.datetimeparts({dtype}, len={LEN}) \
                 child=vortex.dict(i32, len={LEN})\n{expected}"
            )
        );
        Ok(())
    }

    /// With wide seconds DateTimeParts has no op for its days, so days builds no source and
    /// nothing fuses.
    #[test]
    fn wide_seconds_do_not_fuse() -> VortexResult<()> {
        let parts = date_time_parts(false, None)?;
        let parts = parts.as_::<DateTimeParts>();
        let wide = PrimitiveArray::from_iter((0..LEN as i64).map(|i| i % 86_400)).into_array();
        let array = DateTimeParts::try_new(
            parts.dtype().clone(),
            parts.days().clone(),
            wide,
            parts.subseconds().clone(),
        )?
        .into_array();
        let (dtype, days) = (array.dtype().clone(), parts.days().dtype().clone());
        assert_eq!(
            fusion_trace(array, &SESSION)?,
            format!(
                "ExecuteSpine slot=0 parent=vortex.datetimeparts({dtype}, len={LEN}) \
                 child=fastlanes.for.v2({days}, len={LEN})"
            )
        );
        Ok(())
    }

    /// With days already canonical there is nothing to fuse.
    #[test]
    fn canonical_days_decode_level_wise() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let parts = date_time_parts(false, None)?;
        let parts = parts.as_::<DateTimeParts>();
        let days = parts.days().clone().execute::<PrimitiveArray>(&mut ctx)?;
        let array = DateTimeParts::try_new(
            parts.dtype().clone(),
            days.into_array(),
            parts.seconds().clone(),
            parts.subseconds().clone(),
        )?
        .into_array();
        assert_eq!(fusion_trace(array, &SESSION)?, "");
        Ok(())
    }
}

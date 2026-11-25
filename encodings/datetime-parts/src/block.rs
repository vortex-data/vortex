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
/// the time parts, as [`decode_to_temporal`](crate::decode_to_temporal) does.
///
/// `None` when it decodes some other way: a unit of days, or a time part that is neither a
/// non-null constant nor canonical `i32` values, which the level-wise decode casts.
pub fn days_block_op(
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
    fn process<'a>(&mut self, start: usize, block: Block<'a>) -> VortexResult<Written<'a>> {
        let Values::Copy { input, out } = block.values else {
            vortex_bail!("DateTimeParts combines days into a block of its own");
        };
        let (days, len, divisor) = (input.as_slice::<D>(), input.len(), self.divisor);
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
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::block::BlockProgram;
    use vortex_array::block::BlockSource;
    use vortex_array::block::PrimitiveSource;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::match_each_integer_ptype;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_fastlanes::FoR;
    use vortex_fastlanes::bitpack_compress::bitpack_encode;
    use vortex_fastlanes::block::for_block_source;
    use vortex_session::VortexSession;

    use super::days_block_op;
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

    /// The days child as a source: FoR's fused unpack over BitPacked, or its canonical values.
    fn days_source(days: &ArrayRef) -> VortexResult<Box<dyn BlockSource>> {
        let mut ctx = SESSION.create_execution_ctx();
        if let Some(for_) = days.as_opt::<FoR>()
            && let Some(source) = for_block_source(for_, &mut ctx)?
        {
            return Ok(source);
        }
        Ok(PrimitiveSource::boxed(
            &days.clone().execute::<PrimitiveArray>(&mut ctx)?,
        ))
    }

    /// Run DateTimeParts' op over its days child's source, and check it against the storage of
    /// the array decoded level-wise.
    fn assert_combines_like_decode(array: &ArrayRef) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let parts = array.as_::<DateTimeParts>();
        let days = parts.days().clone();
        // Constant time parts stay constant, so the op folds them in; others are executed.
        let mut time_part = |part: &ArrayRef| -> VortexResult<ArrayRef> {
            if part.as_constant().is_some() {
                return Ok(part.clone());
            }
            Ok(part.clone().execute::<Canonical>(&mut ctx)?.into_array())
        };
        let (seconds, subseconds) = (time_part(parts.seconds())?, time_part(parts.subseconds())?);
        let op = days_block_op(array.dtype(), days.dtype(), &seconds, &subseconds)
            .ok_or_else(|| vortex_error::vortex_err!("DateTimeParts has an op"))?;
        let validity = days.validity()?.execute_mask(days.len(), &mut ctx)?;
        let mut program =
            BlockProgram::new(days_source(&days)?, days.dtype().as_ptype(), days.len());
        program.push(op, vortex_array::dtype::PType::I64);
        let output = program.run(&validity, array.dtype().nullability(), &mut ctx)?;
        let expected = array
            .clone()
            .execute::<Canonical>(&mut ctx)?
            .into_extension();
        assert_arrays_eq!(output, expected.storage_array().clone(), &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::plain(false, None)]
    #[case::nullable(true, None)]
    #[case::time_zone(false, Some("Europe/London"))]
    fn days_combine_like_decode(
        #[case] nullable: bool,
        #[case] tz: Option<&str>,
    ) -> VortexResult<()> {
        assert_combines_like_decode(&date_time_parts(nullable, tz)?)
    }

    /// Constant time parts fold into the combine without being materialized.
    #[test]
    fn constant_time_parts() -> VortexResult<()> {
        let parts = date_time_parts(true, None)?;
        let parts = parts.as_::<DateTimeParts>();
        let array = DateTimeParts::try_new(
            parts.dtype().clone(),
            parts.days().clone(),
            ConstantArray::new(Scalar::from(17i32), LEN).into_array(),
            ConstantArray::new(Scalar::from(5i32), LEN).into_array(),
        )?
        .into_array();
        assert_combines_like_decode(&array)
    }

    /// Time parts wider than `i32` are cast by the level-wise decode, so there is no op.
    #[test]
    fn wide_time_parts_have_no_op() -> VortexResult<()> {
        let parts = date_time_parts(false, None)?;
        let parts = parts.as_::<DateTimeParts>();
        let wide = PrimitiveArray::from_iter((0..LEN as i64).map(|i| i % 86_400)).into_array();
        let subseconds = parts.subseconds().clone();
        assert!(days_block_op(parts.dtype(), parts.days().dtype(), &wide, &subseconds).is_none());
        Ok(())
    }
}

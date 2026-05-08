// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical Vortex → Arrow conversion for the temporal extension types.
//!
//! `Date`, `Time`, and `Timestamp` are Vortex builtin extension types that map directly to
//! native Arrow temporal types (`Date32`/`Date64`, `Time32`/`Time64`, `Timestamp`). These
//! conversions live in the canonical executor rather than in the plugin layer because they
//! aren't Arrow extensions and the mapping is fully determined by the source `ExtDType`.

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::PrimitiveArray as ArrowPrimitiveArray;
use arrow_array::types::ArrowTemporalType;
use arrow_array::types::ArrowTimestampType;
use arrow_array::types::Date32Type;
use arrow_array::types::Date64Type;
use arrow_array::types::Time32MillisecondType;
use arrow_array::types::Time32SecondType;
use arrow_array::types::Time64MicrosecondType;
use arrow_array::types::Time64NanosecondType;
use arrow_array::types::TimestampMicrosecondType;
use arrow_array::types::TimestampMillisecondType;
use arrow_array::types::TimestampNanosecondType;
use arrow_array::types::TimestampSecondType;
use arrow_schema::DataType;
use arrow_schema::TimeUnit as ArrowTimeUnit;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::arrow::null_buffer::to_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::extension::datetime::AnyTemporal;
use crate::extension::datetime::TemporalMetadata;
use crate::extension::datetime::TimeUnit;

pub(super) fn to_arrow_date(
    array: ArrayRef,
    target: &DataType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    validate_temporal_extension(&array, target)?;
    Ok(match target {
        DataType::Date32 => Arc::new(to_temporal_primitive::<Date32Type>(array, ctx)?),
        DataType::Date64 => Arc::new(to_temporal_primitive::<Date64Type>(array, ctx)?),
        _ => unreachable!("to_arrow_date called with non-Date type {target}"),
    })
}

pub(super) fn to_arrow_time(
    array: ArrayRef,
    target: &DataType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    validate_temporal_extension(&array, target)?;
    Ok(match target {
        DataType::Time32(ArrowTimeUnit::Second) => {
            Arc::new(to_temporal_primitive::<Time32SecondType>(array, ctx)?)
        }
        DataType::Time32(ArrowTimeUnit::Millisecond) => {
            Arc::new(to_temporal_primitive::<Time32MillisecondType>(array, ctx)?)
        }
        DataType::Time64(ArrowTimeUnit::Microsecond) => {
            Arc::new(to_temporal_primitive::<Time64MicrosecondType>(array, ctx)?)
        }
        DataType::Time64(ArrowTimeUnit::Nanosecond) => {
            Arc::new(to_temporal_primitive::<Time64NanosecondType>(array, ctx)?)
        }
        _ => unreachable!("to_arrow_time called with non-Time type {target}"),
    })
}

pub(super) fn to_arrow_timestamp(
    array: ArrayRef,
    target: &DataType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    validate_temporal_extension(&array, target)?;
    let DataType::Timestamp(unit, tz) = target else {
        unreachable!("to_arrow_timestamp called with non-Timestamp type {target}");
    };
    Ok(match unit {
        ArrowTimeUnit::Second => with_timezone::<TimestampSecondType>(array, tz.as_ref(), ctx)?,
        ArrowTimeUnit::Millisecond => {
            with_timezone::<TimestampMillisecondType>(array, tz.as_ref(), ctx)?
        }
        ArrowTimeUnit::Microsecond => {
            with_timezone::<TimestampMicrosecondType>(array, tz.as_ref(), ctx)?
        }
        ArrowTimeUnit::Nanosecond => {
            with_timezone::<TimestampNanosecondType>(array, tz.as_ref(), ctx)?
        }
    })
}

fn with_timezone<T>(
    array: ArrayRef,
    tz: Option<&Arc<str>>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T: ArrowTimestampType,
    T::Native: NativePType,
{
    Ok(Arc::new(
        to_temporal_primitive::<T>(array, ctx)?.with_timezone_opt(tz.cloned()),
    ))
}

fn to_temporal_primitive<T>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowPrimitiveArray<T>>
where
    T: ArrowTemporalType,
    T::Native: NativePType,
{
    let array = array.cast(DType::Primitive(T::Native::PTYPE, Nullability::Nullable))?;
    let primitive = array.execute::<PrimitiveArray>(ctx)?;
    let validity = primitive
        .as_ref()
        .validity()?
        .execute_mask(primitive.as_ref().len(), ctx)?;
    let buffer = primitive.into_buffer::<T::Native>();
    Ok(ArrowPrimitiveArray::<T>::new(
        buffer.into_arrow_scalar_buffer(),
        to_null_buffer(validity),
    ))
}

/// Verify that the source array is compatible with the target Arrow temporal type.
///
/// Vortex does not silently reinterpret across mismatched temporal units or timezones, so a
/// source with temporal extension metadata must agree exactly with `target`. Non-extension
/// sources are passed through; the cast in `to_temporal_primitive` will surface any width
/// mismatch.
fn validate_temporal_extension(array: &ArrayRef, target: &DataType) -> VortexResult<()> {
    let Some(ext) = array.dtype().as_extension_opt() else {
        return Ok(());
    };
    let Some(temporal) = ext.metadata_opt::<AnyTemporal>() else {
        vortex_bail!(
            "Cannot convert extension {} to Arrow type {target}",
            ext.id()
        );
    };
    match (temporal, target) {
        (TemporalMetadata::Date(TimeUnit::Days), DataType::Date32) => Ok(()),
        (TemporalMetadata::Date(TimeUnit::Milliseconds), DataType::Date64) => Ok(()),
        (
            TemporalMetadata::Time(unit),
            DataType::Time32(arrow_unit) | DataType::Time64(arrow_unit),
        ) if matches!(
            (unit, arrow_unit),
            (TimeUnit::Seconds, ArrowTimeUnit::Second)
                | (TimeUnit::Milliseconds, ArrowTimeUnit::Millisecond)
                | (TimeUnit::Microseconds, ArrowTimeUnit::Microsecond)
                | (TimeUnit::Nanoseconds, ArrowTimeUnit::Nanosecond)
        ) =>
        {
            Ok(())
        }
        (TemporalMetadata::Timestamp(unit, src_tz), DataType::Timestamp(arrow_unit, tgt_tz)) => {
            let src_arrow_unit = ArrowTimeUnit::try_from(*unit)?;
            if src_arrow_unit != *arrow_unit {
                vortex_bail!(
                    "Cannot convert Timestamp({unit}) to Arrow Timestamp({arrow_unit:?}): unit mismatch"
                );
            }
            if src_tz != tgt_tz {
                vortex_bail!(
                    "Cannot convert Timestamp(tz={src_tz:?}) to Arrow Timestamp(tz={tgt_tz:?}): timezone mismatch"
                );
            }
            Ok(())
        }
        (temporal, target) => vortex_bail!(
            "Cannot convert {} to Arrow type {target}",
            match temporal {
                TemporalMetadata::Date(unit) => format!("Date({unit})"),
                TemporalMetadata::Time(unit) => format!("Time({unit})"),
                TemporalMetadata::Timestamp(unit, tz) => format!("Timestamp({unit}, tz={tz:?})"),
            }
        ),
    }
}

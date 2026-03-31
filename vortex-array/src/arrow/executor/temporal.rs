// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::PrimitiveArray;
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
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::ExtensionArray;
use crate::arrays::PrimitiveArray as VortexPrimitiveArray;
use crate::arrow::null_buffer::to_null_buffer;
use crate::dtype::NativePType;
use crate::extension::datetime::AnyTemporal;
use crate::extension::datetime::TemporalMetadata;
use crate::extension::datetime::TimeUnit;

pub(super) fn to_arrow_temporal(
    array: ArrayRef,
    data_type: &DataType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let temporal_options = array
        .dtype()
        .as_extension()
        .metadata_opt::<AnyTemporal>()
        .ok_or_else(|| {
            vortex_err!(
                "Array dtype {} is not a temporal extension type",
                array.dtype()
            )
        })?;

    match (temporal_options, &data_type) {
        (TemporalMetadata::Date(TimeUnit::Days), DataType::Date32) => {
            to_temporal::<Date32Type>(array, ctx)
        }
        (TemporalMetadata::Date(TimeUnit::Milliseconds), DataType::Date64) => {
            to_temporal::<Date64Type>(array, ctx)
        }
        (TemporalMetadata::Time(TimeUnit::Seconds), DataType::Time32(ArrowTimeUnit::Second)) => {
            to_temporal::<Time32SecondType>(array, ctx)
        }
        (
            TemporalMetadata::Time(TimeUnit::Milliseconds),
            DataType::Time32(ArrowTimeUnit::Millisecond),
        ) => to_temporal::<Time32MillisecondType>(array, ctx),
        (
            TemporalMetadata::Time(TimeUnit::Microseconds),
            DataType::Time64(ArrowTimeUnit::Microsecond),
        ) => to_temporal::<Time64MicrosecondType>(array, ctx),

        (
            TemporalMetadata::Time(TimeUnit::Nanoseconds),
            DataType::Time64(ArrowTimeUnit::Nanosecond),
        ) => to_temporal::<Time64NanosecondType>(array, ctx),

        (TemporalMetadata::Timestamp(unit, tz), DataType::Timestamp(arrow_unit, arrow_tz)) => {
            vortex_ensure!(
                tz == arrow_tz,
                "Cannot convert {} array to Arrow type {} due to timezone mismatch",
                array.dtype(),
                data_type
            );

            match (unit, arrow_unit) {
                (TimeUnit::Seconds, ArrowTimeUnit::Second) => {
                    to_arrow_timestamp::<TimestampSecondType>(array, arrow_tz, ctx)
                }
                (TimeUnit::Milliseconds, ArrowTimeUnit::Millisecond) => {
                    to_arrow_timestamp::<TimestampMillisecondType>(array, arrow_tz, ctx)
                }
                (TimeUnit::Microseconds, ArrowTimeUnit::Microsecond) => {
                    to_arrow_timestamp::<TimestampMicrosecondType>(array, arrow_tz, ctx)
                }
                (TimeUnit::Nanoseconds, ArrowTimeUnit::Nanosecond) => {
                    to_arrow_timestamp::<TimestampNanosecondType>(array, arrow_tz, ctx)
                }
                _ => vortex_bail!(
                    "Cannot convert {} array to Arrow type {}",
                    array.dtype(),
                    data_type
                ),
            }
        }
        _ => vortex_bail!(
            "Cannot convert {} array to Arrow type {}",
            array.dtype(),
            data_type
        ),
    }
}

fn to_temporal<T: ArrowTemporalType>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T::Native: NativePType,
{
    // We cast the array to the native primitive type.
    Ok(Arc::new(to_arrow_temporal_primitive::<T>(array, ctx)?))
}

fn to_arrow_timestamp<T: ArrowTimestampType>(
    array: ArrayRef,
    arrow_tz: &Option<Arc<str>>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T::Native: NativePType,
{
    Ok(Arc::new(
        to_arrow_temporal_primitive::<T>(array, ctx)?.with_timezone_opt(arrow_tz.clone()),
    ))
}

fn to_arrow_temporal_primitive<T: ArrowTemporalType>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowPrimitiveArray<T>>
where
    T::Native: NativePType,
{
    debug_assert!(array.dtype().as_extension().is::<AnyTemporal>());

    let ext_array = array.execute::<ExtensionArray>(ctx)?;
    let primitive = ext_array
        .storage_array()
        .clone()
        .execute::<VortexPrimitiveArray>(ctx)?;
    vortex_ensure!(
        primitive.ptype() == T::Native::PTYPE,
        "Expected temporal array to produce vector of width {}, found {}",
        T::Native::PTYPE,
        primitive.ptype()
    );

    let validity = primitive.validity_mask()?;
    let buffer = primitive.to_buffer::<T::Native>();

    let values = buffer.into_arrow_scalar_buffer();
    let nulls = to_null_buffer(validity);

    Ok(PrimitiveArray::<T>::new(values, nulls))
}

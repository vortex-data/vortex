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
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::PTypeDowncastExt;
use vortex_dtype::datetime::TemporalMetadata;
use vortex_dtype::datetime::TimeUnit;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::VectorExecutor;
use crate::arrow::null_buffer::to_null_buffer;

pub(super) fn to_arrow_temporal(
    array: ArrayRef,
    data_type: &DataType,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    let DType::Extension(ext_dtype) = array.dtype() else {
        vortex_bail!(
            "Cannot convert array with DType {} to temporal Arrow type",
            array.dtype()
        );
    };

    let temporal_metadata = TemporalMetadata::try_from(ext_dtype)?;

    match (temporal_metadata, &data_type) {
        (TemporalMetadata::Date(TimeUnit::Days), DataType::Date32) => {
            to_temporal::<Date32Type>(array, session)
        }
        (TemporalMetadata::Date(TimeUnit::Milliseconds), DataType::Date64) => {
            to_temporal::<Date64Type>(array, session)
        }
        (TemporalMetadata::Time(TimeUnit::Seconds), DataType::Time32(ArrowTimeUnit::Second)) => {
            to_temporal::<Time32SecondType>(array, session)
        }
        (
            TemporalMetadata::Time(TimeUnit::Milliseconds),
            DataType::Time32(ArrowTimeUnit::Millisecond),
        ) => to_temporal::<Time32MillisecondType>(array, session),
        (
            TemporalMetadata::Time(TimeUnit::Microseconds),
            DataType::Time64(ArrowTimeUnit::Microsecond),
        ) => to_temporal::<Time64MicrosecondType>(array, session),

        (
            TemporalMetadata::Time(TimeUnit::Nanoseconds),
            DataType::Time64(ArrowTimeUnit::Nanosecond),
        ) => to_temporal::<Time64NanosecondType>(array, session),
        (
            TemporalMetadata::Timestamp(TimeUnit::Seconds, _),
            DataType::Timestamp(ArrowTimeUnit::Second, arrow_tz),
        ) => to_arrow_timestamp::<TimestampSecondType>(array, arrow_tz, session),
        (
            TemporalMetadata::Timestamp(TimeUnit::Milliseconds, _),
            DataType::Timestamp(ArrowTimeUnit::Millisecond, arrow_tz),
        ) => to_arrow_timestamp::<TimestampMillisecondType>(array, arrow_tz, session),
        (
            TemporalMetadata::Timestamp(TimeUnit::Microseconds, _),
            DataType::Timestamp(ArrowTimeUnit::Microsecond, arrow_tz),
        ) => to_arrow_timestamp::<TimestampMicrosecondType>(array, arrow_tz, session),
        (
            TemporalMetadata::Timestamp(TimeUnit::Nanoseconds, _),
            DataType::Timestamp(ArrowTimeUnit::Nanosecond, arrow_tz),
        ) => to_arrow_timestamp::<TimestampNanosecondType>(array, arrow_tz, session),
        _ => vortex_bail!(
            "Cannot convert {} array to Arrow type {}",
            array.dtype(),
            data_type
        ),
    }
}

fn to_temporal<T: ArrowTemporalType>(
    array: ArrayRef,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef>
where
    T::Native: NativePType,
{
    // We cast the array to the native primitive type.
    Ok(Arc::new(to_arrow_temporal_primitive::<T>(array, session)?))
}

fn to_arrow_timestamp<T: ArrowTimestampType>(
    array: ArrayRef,
    arrow_tz: &Option<Arc<str>>,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef>
where
    T::Native: NativePType,
{
    Ok(Arc::new(
        to_arrow_temporal_primitive::<T>(array, session)?.with_timezone_opt(arrow_tz.clone()),
    ))
}

fn to_arrow_temporal_primitive<T: ArrowTemporalType>(
    array: ArrayRef,
    session: &VortexSession,
) -> VortexResult<ArrowPrimitiveArray<T>>
where
    T::Native: NativePType,
{
    let vector = array.execute_vector(session)?.into_primitive();
    vortex_ensure!(
        vector.ptype() == T::Native::PTYPE,
        "Expected temporal array to produce vector of width {}, found {}",
        T::Native::PTYPE,
        vector.ptype()
    );

    let (buffer, validity) = vector.downcast::<T::Native>().into_parts();

    let values = buffer.into_arrow_scalar_buffer();
    let nulls = to_null_buffer(validity);

    Ok(PrimitiveArray::<T>::new(values, nulls))
}

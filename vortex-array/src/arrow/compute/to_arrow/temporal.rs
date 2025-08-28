// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::types::{
    ArrowTemporalType, ArrowTimestampType, Date32Type, Date64Type, Time32MillisecondType,
    Time32SecondType, Time64MicrosecondType, Time64NanosecondType, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
};
use arrow_array::{ArrayRef as ArrowArrayRef, PrimitiveArray as ArrowPrimitiveArray};
use arrow_schema::{DataType, TimeUnit as ArrowTimeUnit};
use vortex_dtype::datetime::{TemporalMetadata, TimeUnit, is_temporal_ext_type};
use vortex_dtype::{DType, NativePType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::arrays::{ExtensionVTable, TemporalArray};
use crate::arrow::array::ArrowArray;
use crate::arrow::compute::to_arrow::ToArrowArgs;
use crate::compute::{InvocationArgs, Kernel, Output, cast};
use crate::{Array as _, IntoArray, ToCanonical};

/// Implementation of `ToArrow` kernel for canonical Vortex arrays.
#[derive(Debug)]
pub(super) struct ToArrowTemporal;

impl Kernel for ToArrowTemporal {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let ToArrowArgs { array, arrow_type } = ToArrowArgs::try_from(args)?;

        if !array
            .as_opt::<ExtensionVTable>()
            .is_some_and(|ext| is_temporal_ext_type(ext.ext_dtype().id()))
        {
            // This kernel only handles temporal arrays.
            return Ok(None);
        }
        let array = TemporalArray::try_from(array.to_array())
            .vortex_expect("Checked above that array is a temporal ExtensionArray");

        // Figure out the target Arrow type, or use the canonical type
        let arrow_type = arrow_type
            .cloned()
            .map(Ok)
            .unwrap_or_else(|| array.dtype().to_arrow_dtype())?;

        let arrow_array: ArrowArrayRef = match (array.temporal_metadata(), &arrow_type) {
            (TemporalMetadata::Date(TimeUnit::Days), DataType::Date32) => {
                to_arrow_temporal::<Date32Type>(&array)
            }
            (TemporalMetadata::Date(TimeUnit::Milliseconds), DataType::Date64) => {
                to_arrow_temporal::<Date64Type>(&array)
            }
            (
                TemporalMetadata::Time(TimeUnit::Seconds),
                DataType::Time32(ArrowTimeUnit::Second),
            ) => to_arrow_temporal::<Time32SecondType>(&array),
            (
                TemporalMetadata::Time(TimeUnit::Milliseconds),
                DataType::Time32(ArrowTimeUnit::Millisecond),
            ) => to_arrow_temporal::<Time32MillisecondType>(&array),
            (
                TemporalMetadata::Time(TimeUnit::Microseconds),
                DataType::Time64(ArrowTimeUnit::Microsecond),
            ) => to_arrow_temporal::<Time64MicrosecondType>(&array),

            (
                TemporalMetadata::Time(TimeUnit::Nanoseconds),
                DataType::Time64(ArrowTimeUnit::Nanosecond),
            ) => to_arrow_temporal::<Time64NanosecondType>(&array),
            (
                TemporalMetadata::Timestamp(TimeUnit::Seconds, _),
                DataType::Timestamp(ArrowTimeUnit::Second, arrow_tz),
            ) => to_arrow_timestamp::<TimestampSecondType>(&array, arrow_tz),
            (
                TemporalMetadata::Timestamp(TimeUnit::Milliseconds, _),
                DataType::Timestamp(ArrowTimeUnit::Millisecond, arrow_tz),
            ) => to_arrow_timestamp::<TimestampMillisecondType>(&array, arrow_tz),
            (
                TemporalMetadata::Timestamp(TimeUnit::Microseconds, _),
                DataType::Timestamp(ArrowTimeUnit::Microsecond, arrow_tz),
            ) => to_arrow_timestamp::<TimestampMicrosecondType>(&array, arrow_tz),
            (
                TemporalMetadata::Timestamp(TimeUnit::Nanoseconds, _),
                DataType::Timestamp(ArrowTimeUnit::Nanosecond, arrow_tz),
            ) => to_arrow_timestamp::<TimestampNanosecondType>(&array, arrow_tz),
            _ => vortex_bail!(
                "Cannot convert {} array to Arrow type {}",
                array.dtype(),
                arrow_type,
            ),
        }?;

        Ok(Some(
            ArrowArray::new(arrow_array, array.dtype().nullability())
                .into_array()
                .into(),
        ))
    }
}

fn to_arrow_temporal_primitive<T: ArrowTemporalType>(
    array: &TemporalArray,
) -> VortexResult<ArrowPrimitiveArray<T>>
where
    T::Native: NativePType,
{
    let values_dtype = DType::Primitive(T::Native::PTYPE, array.dtype().nullability());
    let values = cast(array.temporal_values(), &values_dtype)?
        .to_primitive()?
        .into_buffer()
        .into_arrow_scalar_buffer();
    let nulls = array.temporal_values().validity_mask()?.to_null_buffer();
    Ok(ArrowPrimitiveArray::<T>::new(values, nulls))
}

fn to_arrow_temporal<T: ArrowTemporalType>(array: &TemporalArray) -> VortexResult<ArrowArrayRef>
where
    T::Native: NativePType,
{
    Ok(Arc::new(to_arrow_temporal_primitive::<T>(array)?))
}

fn to_arrow_timestamp<T: ArrowTimestampType>(
    array: &TemporalArray,
    arrow_tz: &Option<Arc<str>>,
) -> VortexResult<ArrowArrayRef>
where
    T::Native: NativePType,
{
    Ok(Arc::new(
        to_arrow_temporal_primitive::<T>(array)?.with_timezone_opt(arrow_tz.clone()),
    ))
}

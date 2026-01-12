// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::PTypeDowncastExt;
use vortex_dtype::datetime::TemporalMetadata;
use vortex_dtype::datetime::TimeUnit;
use vortex_dtype::datetime::is_temporal_ext_type;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VectorExecutor;
use crate::VortexSessionExecute;
use crate::arrays::ExtensionVTable;
use crate::arrays::TemporalArray;
use crate::arrow::array::ArrowArray;
use crate::arrow::compute::to_arrow::ToArrowArgs;
use crate::arrow::null_buffer::to_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::compute::InvocationArgs;
use crate::compute::Kernel;
use crate::compute::Output;

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
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let values = array
        .temporal_values()
        .cast(values_dtype)?
        .execute(&mut ctx)?
        .to_vector(&mut ctx)?
        .into_primitive()
        .downcast::<T::Native>();

    let (buffer, validity) = values.into_parts();
    let values = buffer.into_arrow_scalar_buffer();
    let nulls = to_null_buffer(validity);

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

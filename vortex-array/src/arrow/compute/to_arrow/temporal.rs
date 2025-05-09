use std::sync::Arc;

use arrow_array::types::{
    Date32Type, Date64Type, Time32MillisecondType, Time32SecondType, Time64MicrosecondType,
    Time64NanosecondType, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType,
};
use arrow_array::{
    ArrayRef as ArrowArrayRef, ArrowPrimitiveType, PrimitiveArray as ArrowPrimitiveArray,
};
use arrow_schema::{DataType, TimeUnit as ArrowTimeUnit};
use vortex_dtype::datetime::{TemporalMetadata, TimeUnit, is_temporal_ext_type};
use vortex_dtype::{DType, NativePType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::arrays::{ExtensionArray, TemporalArray};
use crate::arrow::array::ArrowArray;
use crate::arrow::compute::to_arrow::ToArrowArgs;
use crate::compute::{InvocationArgs, Kernel, Output, cast};
use crate::{Array as _, ToCanonical};

/// Implementation of `ToArrow` kernel for canonical Vortex arrays.
#[derive(Debug)]
pub(super) struct ToArrowTemporal;

impl Kernel for ToArrowTemporal {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let ToArrowArgs { array, arrow_type } = ToArrowArgs::try_from(args)?;

        if !array
            .as_any()
            .downcast_ref::<ExtensionArray>()
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
            (TemporalMetadata::Date(TimeUnit::D), DataType::Date32) => {
                to_arrow_temporal::<Date32Type>(&array)
            }
            (TemporalMetadata::Date(TimeUnit::Ms), DataType::Date64) => {
                to_arrow_temporal::<Date64Type>(&array)
            }
            (TemporalMetadata::Time(TimeUnit::S), DataType::Time32(ArrowTimeUnit::Second)) => {
                to_arrow_temporal::<Time32SecondType>(&array)
            }
            (
                TemporalMetadata::Time(TimeUnit::Ms),
                DataType::Time32(ArrowTimeUnit::Millisecond),
            ) => to_arrow_temporal::<Time32MillisecondType>(&array),
            (
                TemporalMetadata::Time(TimeUnit::Us),
                DataType::Time64(ArrowTimeUnit::Microsecond),
            ) => to_arrow_temporal::<Time64MicrosecondType>(&array),

            (TemporalMetadata::Time(TimeUnit::Ns), DataType::Time64(ArrowTimeUnit::Nanosecond)) => {
                to_arrow_temporal::<Time64NanosecondType>(&array)
            }
            (
                TemporalMetadata::Timestamp(TimeUnit::S, _),
                DataType::Timestamp(ArrowTimeUnit::Second, None),
            ) => to_arrow_temporal::<TimestampSecondType>(&array),
            (
                TemporalMetadata::Timestamp(TimeUnit::Ms, _),
                DataType::Timestamp(ArrowTimeUnit::Millisecond, None),
            ) => to_arrow_temporal::<TimestampMillisecondType>(&array),
            (
                TemporalMetadata::Timestamp(TimeUnit::Us, _),
                DataType::Timestamp(ArrowTimeUnit::Microsecond, None),
            ) => to_arrow_temporal::<TimestampMicrosecondType>(&array),
            (
                TemporalMetadata::Timestamp(TimeUnit::Ns, _),
                DataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
            ) => to_arrow_temporal::<TimestampNanosecondType>(&array),
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

fn to_arrow_temporal<T: ArrowPrimitiveType>(array: &TemporalArray) -> VortexResult<ArrowArrayRef>
where
    T::Native: NativePType,
{
    let values_dtype = DType::Primitive(T::Native::PTYPE, array.dtype().nullability());
    let values = cast(array.temporal_values(), &values_dtype)?
        .to_primitive()?
        .into_buffer()
        .into_arrow_scalar_buffer();
    let nulls = array.temporal_values().validity_mask()?.to_null_buffer();

    Ok(Arc::new(ArrowPrimitiveArray::<T>::new(values, nulls)))
}

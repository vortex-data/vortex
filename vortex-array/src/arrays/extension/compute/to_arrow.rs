use std::sync::Arc;

use arrow_array::{
    ArrayRef, Date32Array, Date64Array, Time32MillisecondArray, Time32SecondArray,
    Time64MicrosecondArray, Time64NanosecondArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
};
use arrow_schema::DataType;
use vortex_dtype::datetime::{TemporalMetadata, TimeUnit, is_temporal_ext_type};
use vortex_dtype::{DType, NativePType};
use vortex_error::{VortexResult, vortex_bail};

use crate::Array;
use crate::arrays::{ExtensionArray, ExtensionEncoding, TemporalArray};
use crate::canonical::ToCanonical;
use crate::compute::{ToArrowFn, cast, to_arrow};

impl ToArrowFn<&ExtensionArray> for ExtensionEncoding {
    fn to_arrow(
        &self,
        array: &ExtensionArray,
        data_type: &DataType,
    ) -> VortexResult<Option<ArrayRef>> {
        // NOTE(ngates): this is really gross... but I guess it's ok given how tightly integrated
        //  we are with Arrow.
        if is_temporal_ext_type(array.id()) {
            temporal_to_arrow(TemporalArray::try_from(array.to_array())?).map(Some)
        } else {
            // Convert storage array directly into arrow, losing type information
            // that will let us round-trip.
            // TODO(aduffy): https://github.com/spiraldb/vortex/issues/1167
            to_arrow(array.storage(), data_type).map(Some)
        }
    }
}

fn temporal_to_arrow(temporal_array: TemporalArray) -> VortexResult<ArrayRef> {
    macro_rules! extract_temporal_values {
        ($values:expr, $prim:ty) => {{
            let temporal_values = cast(
                $values,
                &DType::Primitive(<$prim as NativePType>::PTYPE, $values.dtype().nullability()),
            )?
            .to_primitive()?;
            let nulls = temporal_values.validity_mask()?.to_null_buffer();
            let scalars = temporal_values.into_buffer().into_arrow_scalar_buffer();

            (scalars, nulls)
        }};
    }

    Ok(match temporal_array.temporal_metadata() {
        TemporalMetadata::Date(time_unit) => match time_unit {
            TimeUnit::D => {
                let (scalars, nulls) =
                    extract_temporal_values!(temporal_array.temporal_values(), i32);
                Arc::new(Date32Array::new(scalars, nulls))
            }
            TimeUnit::Ms => {
                let (scalars, nulls) =
                    extract_temporal_values!(temporal_array.temporal_values(), i64);
                Arc::new(Date64Array::new(scalars, nulls))
            }
            _ => vortex_bail!(
                "Invalid TimeUnit {time_unit} for {}",
                temporal_array.ext_dtype().id()
            ),
        },
        TemporalMetadata::Time(time_unit) => match time_unit {
            TimeUnit::S => {
                let (scalars, nulls) =
                    extract_temporal_values!(temporal_array.temporal_values(), i32);
                Arc::new(Time32SecondArray::new(scalars, nulls))
            }
            TimeUnit::Ms => {
                let (scalars, nulls) =
                    extract_temporal_values!(temporal_array.temporal_values(), i32);
                Arc::new(Time32MillisecondArray::new(scalars, nulls))
            }
            TimeUnit::Us => {
                let (scalars, nulls) =
                    extract_temporal_values!(temporal_array.temporal_values(), i64);
                Arc::new(Time64MicrosecondArray::new(scalars, nulls))
            }
            TimeUnit::Ns => {
                let (scalars, nulls) =
                    extract_temporal_values!(temporal_array.temporal_values(), i64);
                Arc::new(Time64NanosecondArray::new(scalars, nulls))
            }
            _ => vortex_bail!(
                "Invalid TimeUnit {time_unit} for {}",
                temporal_array.ext_dtype().id()
            ),
        },
        TemporalMetadata::Timestamp(time_unit, _) => {
            let (scalars, nulls) = extract_temporal_values!(temporal_array.temporal_values(), i64);
            match time_unit {
                TimeUnit::Ns => Arc::new(TimestampNanosecondArray::new(scalars, nulls)),
                TimeUnit::Us => Arc::new(TimestampMicrosecondArray::new(scalars, nulls)),
                TimeUnit::Ms => Arc::new(TimestampMillisecondArray::new(scalars, nulls)),
                TimeUnit::S => Arc::new(TimestampSecondArray::new(scalars, nulls)),
                _ => vortex_bail!(
                    "Invalid TimeUnit {time_unit} for {}",
                    temporal_array.ext_dtype().id()
                ),
            }
        }
    })
}

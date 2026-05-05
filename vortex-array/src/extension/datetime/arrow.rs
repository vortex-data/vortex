// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow export plugins for the built-in temporal extension types.

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::PrimitiveArray;
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
use crate::arrays::ExtensionArray;
use crate::arrays::PrimitiveArray as VortexPrimitiveArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrow::export_plugin::ArrowExportPlugin;
use crate::arrow::to_null_buffer;
use crate::dtype::NativePType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::executor::ExecutionCtx;
use crate::extension::datetime::Date;
use crate::extension::datetime::Time;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;

/// Arrow export plugin for [`Date`] extension types.
#[derive(Debug)]
pub struct DateArrowExport;

impl ArrowExportPlugin for DateArrowExport {
    fn id(&self) -> ExtId {
        Date.id()
    }

    fn to_arrow_data_type(&self, ext_dtype: &ExtDTypeRef) -> VortexResult<DataType> {
        let unit = date_metadata(ext_dtype)?;
        date_arrow_type(unit)
    }

    fn execute_to_arrow(
        &self,
        array: ArrayRef,
        target: &DataType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        match target {
            DataType::Date32 => to_temporal::<Date32Type>(array, ctx),
            DataType::Date64 => to_temporal::<Date64Type>(array, ctx),
            _ => vortex_bail!(
                "Cannot convert {} array to Arrow type {target}",
                array.dtype()
            ),
        }
    }
}

/// Arrow export plugin for [`Time`] extension types.
#[derive(Debug)]
pub struct TimeArrowExport;

impl ArrowExportPlugin for TimeArrowExport {
    fn id(&self) -> ExtId {
        Time.id()
    }

    fn to_arrow_data_type(&self, ext_dtype: &ExtDTypeRef) -> VortexResult<DataType> {
        let unit = time_metadata(ext_dtype)?;
        time_arrow_type(unit)
    }

    fn execute_to_arrow(
        &self,
        array: ArrayRef,
        target: &DataType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        match target {
            DataType::Time32(ArrowTimeUnit::Second) => to_temporal::<Time32SecondType>(array, ctx),
            DataType::Time32(ArrowTimeUnit::Millisecond) => {
                to_temporal::<Time32MillisecondType>(array, ctx)
            }
            DataType::Time64(ArrowTimeUnit::Microsecond) => {
                to_temporal::<Time64MicrosecondType>(array, ctx)
            }
            DataType::Time64(ArrowTimeUnit::Nanosecond) => {
                to_temporal::<Time64NanosecondType>(array, ctx)
            }
            _ => vortex_bail!(
                "Cannot convert {} array to Arrow type {target}",
                array.dtype()
            ),
        }
    }
}

/// Arrow export plugin for [`Timestamp`] extension types.
#[derive(Debug)]
pub struct TimestampArrowExport;

impl ArrowExportPlugin for TimestampArrowExport {
    fn id(&self) -> ExtId {
        Timestamp.id()
    }

    fn to_arrow_data_type(&self, ext_dtype: &ExtDTypeRef) -> VortexResult<DataType> {
        let opts = ext_dtype
            .metadata_opt::<Timestamp>()
            .ok_or_else(|| vortex_err!("expected Timestamp metadata, got {}", ext_dtype.id()))?;
        Ok(DataType::Timestamp(
            ArrowTimeUnit::try_from(opts.unit)?,
            opts.tz.clone(),
        ))
    }

    fn execute_to_arrow(
        &self,
        array: ArrayRef,
        target: &DataType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        let DataType::Timestamp(arrow_unit, arrow_tz) = target else {
            vortex_bail!(
                "Cannot convert {} array to Arrow type {target}",
                array.dtype()
            );
        };

        let opts = array
            .dtype()
            .as_extension()
            .metadata_opt::<Timestamp>()
            .ok_or_else(|| vortex_err!("expected Timestamp metadata, got {}", array.dtype()))?;

        vortex_ensure!(
            &opts.tz == arrow_tz,
            "Cannot convert {} array to Arrow type {} due to timezone mismatch",
            array.dtype(),
            target
        );

        match (&opts.unit, arrow_unit) {
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
                target
            ),
        }
    }
}

fn date_metadata(ext_dtype: &ExtDTypeRef) -> VortexResult<&TimeUnit> {
    ext_dtype
        .metadata_opt::<Date>()
        .ok_or_else(|| vortex_err!("expected Date metadata, got {}", ext_dtype.id()))
}

fn time_metadata(ext_dtype: &ExtDTypeRef) -> VortexResult<&TimeUnit> {
    ext_dtype
        .metadata_opt::<Time>()
        .ok_or_else(|| vortex_err!("expected Time metadata, got {}", ext_dtype.id()))
}

fn date_arrow_type(unit: &TimeUnit) -> VortexResult<DataType> {
    Ok(match unit {
        TimeUnit::Days => DataType::Date32,
        TimeUnit::Milliseconds => DataType::Date64,
        TimeUnit::Nanoseconds | TimeUnit::Microseconds | TimeUnit::Seconds => {
            vortex_bail!("Date does not support time unit {unit}")
        }
    })
}

fn time_arrow_type(unit: &TimeUnit) -> VortexResult<DataType> {
    Ok(match unit {
        TimeUnit::Seconds => DataType::Time32(ArrowTimeUnit::Second),
        TimeUnit::Milliseconds => DataType::Time32(ArrowTimeUnit::Millisecond),
        TimeUnit::Microseconds => DataType::Time64(ArrowTimeUnit::Microsecond),
        TimeUnit::Nanoseconds => DataType::Time64(ArrowTimeUnit::Nanosecond),
        TimeUnit::Days => vortex_bail!("Time does not support time unit {unit}"),
    })
}

fn to_temporal<T: ArrowTemporalType>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T::Native: NativePType,
{
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
) -> VortexResult<PrimitiveArray<T>>
where
    T::Native: NativePType,
{
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

    let validity = primitive
        .as_ref()
        .validity()?
        .to_mask(primitive.as_ref().len(), ctx)?;
    let buffer = primitive.to_buffer::<T::Native>();

    let values = buffer.into_arrow_scalar_buffer();
    let nulls = to_null_buffer(validity);

    Ok(PrimitiveArray::<T>::new(values, nulls))
}

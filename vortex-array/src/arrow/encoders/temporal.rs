// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporal-extension Arrow plugins for `vortex.date`, `vortex.time`, and `vortex.timestamp`.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::TimeUnit as ArrowTimeUnit;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrow::ArrowEncoder;
use crate::arrow::ArrowSession;
use crate::arrow::dtype_converter::ArrowDTypeConverter;
use crate::arrow::executor::temporal::to_arrow_temporal;
use crate::dtype::DType;
use crate::dtype::extension::ExtDTypeRef;
use crate::extension::datetime::AnyTemporal;
use crate::extension::datetime::TemporalMetadata;
use crate::extension::datetime::TimeUnit;

/// Map a temporal extension dtype to its preferred Arrow [`DataType`].
fn temporal_arrow_data_type(ext: &ExtDTypeRef) -> VortexResult<DataType> {
    let temporal = ext
        .metadata_opt::<AnyTemporal>()
        .ok_or_else(|| vortex_err!("ExtDType {} is not a temporal extension", ext.id()))?;
    Ok(match temporal {
        TemporalMetadata::Timestamp(unit, tz) => {
            DataType::Timestamp(ArrowTimeUnit::try_from(*unit)?, tz.clone())
        }
        TemporalMetadata::Date(unit) => match unit {
            TimeUnit::Days => DataType::Date32,
            TimeUnit::Milliseconds => DataType::Date64,
            TimeUnit::Nanoseconds | TimeUnit::Microseconds | TimeUnit::Seconds => {
                vortex_panic!(InvalidArgument: "Invalid TimeUnit {} for {}", unit, ext.id())
            }
        },
        TemporalMetadata::Time(unit) => match unit {
            TimeUnit::Seconds => DataType::Time32(ArrowTimeUnit::Second),
            TimeUnit::Milliseconds => DataType::Time32(ArrowTimeUnit::Millisecond),
            TimeUnit::Microseconds => DataType::Time64(ArrowTimeUnit::Microsecond),
            TimeUnit::Nanoseconds => DataType::Time64(ArrowTimeUnit::Nanosecond),
            TimeUnit::Days => {
                vortex_panic!(InvalidArgument: "Invalid TimeUnit {} for {}", unit, ext.id())
            }
        },
    })
}

/// [`ArrowDTypeConverter`] for the built-in temporal extensions
/// (`vortex.date`, `vortex.time`, `vortex.timestamp`).
#[derive(Debug, Default)]
pub struct TemporalArrowDTypeConverter;

impl ArrowDTypeConverter for TemporalArrowDTypeConverter {
    fn to_arrow_data_type(&self, ext: &ExtDTypeRef) -> VortexResult<DataType> {
        temporal_arrow_data_type(ext)
    }
}

/// [`ArrowEncoder`] for the built-in temporal extensions. Registered against the
/// `vortex.date`, `vortex.time`, and `vortex.timestamp` [`crate::dtype::extension::ExtId`]s.
#[derive(Debug, Default)]
pub struct TemporalArrowEncoder;

impl ArrowEncoder for TemporalArrowEncoder {
    fn preferred_arrow_type(
        &self,
        array: &ArrayRef,
        _session: &ArrowSession,
    ) -> VortexResult<Option<DataType>> {
        let DType::Extension(ext) = array.dtype() else {
            return Ok(None);
        };
        if ext.metadata_opt::<AnyTemporal>().is_none() {
            return Ok(None);
        }
        Ok(Some(temporal_arrow_data_type(ext)?))
    }

    fn to_arrow_array(
        &self,
        array: ArrayRef,
        target: &DataType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        match target {
            DataType::Timestamp(..)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_) => to_arrow_temporal(array, target, ctx).map(Some),
            _ => Ok(None),
        }
    }
}

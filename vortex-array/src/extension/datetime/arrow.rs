// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrowVTable`] impls for the temporal extension types.

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
use arrow_schema::Field;
use arrow_schema::TimeUnit as ArrowTimeUnit;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::ExtensionArray;
use crate::arrays::PrimitiveArray as VortexPrimitiveArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrow::ArrowVTable;
use crate::arrow::to_null_buffer;
use crate::dtype::NativePType;
use crate::dtype::extension::ExtId;
use crate::extension::datetime::Date;
use crate::extension::datetime::Time;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;

impl ArrowVTable for Date {
    fn vortex_ext_id(&self) -> ExtId {
        Date.id()
    }

    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &crate::dtype::DType,
        _session: &VortexSession,
    ) -> VortexResult<Field> {
        let unit = unit_for::<Date>(dtype)?;
        let data_type = match unit {
            TimeUnit::Days => DataType::Date32,
            TimeUnit::Milliseconds => DataType::Date64,
            other => vortex_bail!("Date does not support time unit {other}"),
        };
        Ok(Field::new(name, data_type, dtype.is_nullable()))
    }

    fn from_arrow_field(&self, field: &Field) -> VortexResult<crate::dtype::DType> {
        let unit = match field.data_type() {
            DataType::Date32 => TimeUnit::Days,
            DataType::Date64 => TimeUnit::Milliseconds,
            other => vortex_bail!("Date plugin cannot convert Arrow type {other}"),
        };
        Ok(crate::dtype::DType::Extension(
            Date::new(unit, field.is_nullable().into()).erased(),
        ))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        match (unit_for::<Date>(array.dtype())?, target.data_type()) {
            (TimeUnit::Days, DataType::Date32) => to_temporal_array::<Date32Type>(array, ctx),
            (TimeUnit::Milliseconds, DataType::Date64) => {
                to_temporal_array::<Date64Type>(array, ctx)
            }
            (unit, dt) => vortex_bail!("Cannot convert Date({unit}) array to Arrow type {dt}"),
        }
    }

    fn from_arrow_array(&self, _array: ArrowArrayRef, _field: &Field) -> VortexResult<ArrayRef> {
        vortex_bail!("Date::from_arrow_array is not yet implemented")
    }
}

impl ArrowVTable for Time {
    fn vortex_ext_id(&self) -> ExtId {
        Time.id()
    }

    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &crate::dtype::DType,
        _session: &VortexSession,
    ) -> VortexResult<Field> {
        let unit = unit_for::<Time>(dtype)?;
        let data_type = match unit {
            TimeUnit::Seconds => DataType::Time32(ArrowTimeUnit::Second),
            TimeUnit::Milliseconds => DataType::Time32(ArrowTimeUnit::Millisecond),
            TimeUnit::Microseconds => DataType::Time64(ArrowTimeUnit::Microsecond),
            TimeUnit::Nanoseconds => DataType::Time64(ArrowTimeUnit::Nanosecond),
            TimeUnit::Days => vortex_bail!("Time does not support time unit Days"),
        };
        Ok(Field::new(name, data_type, dtype.is_nullable()))
    }

    fn from_arrow_field(&self, field: &Field) -> VortexResult<crate::dtype::DType> {
        let unit = match field.data_type() {
            DataType::Time32(u) | DataType::Time64(u) => TimeUnit::from(*u),
            other => vortex_bail!("Time plugin cannot convert Arrow type {other}"),
        };
        Ok(crate::dtype::DType::Extension(
            Time::new(unit, field.is_nullable().into()).erased(),
        ))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        match (unit_for::<Time>(array.dtype())?, target.data_type()) {
            (TimeUnit::Seconds, DataType::Time32(ArrowTimeUnit::Second)) => {
                to_temporal_array::<Time32SecondType>(array, ctx)
            }
            (TimeUnit::Milliseconds, DataType::Time32(ArrowTimeUnit::Millisecond)) => {
                to_temporal_array::<Time32MillisecondType>(array, ctx)
            }
            (TimeUnit::Microseconds, DataType::Time64(ArrowTimeUnit::Microsecond)) => {
                to_temporal_array::<Time64MicrosecondType>(array, ctx)
            }
            (TimeUnit::Nanoseconds, DataType::Time64(ArrowTimeUnit::Nanosecond)) => {
                to_temporal_array::<Time64NanosecondType>(array, ctx)
            }
            (unit, dt) => vortex_bail!("Cannot convert Time({unit}) array to Arrow type {dt}"),
        }
    }

    fn from_arrow_array(&self, _array: ArrowArrayRef, _field: &Field) -> VortexResult<ArrayRef> {
        vortex_bail!("Time::from_arrow_array is not yet implemented")
    }
}

impl ArrowVTable for Timestamp {
    fn vortex_ext_id(&self) -> ExtId {
        Timestamp.id()
    }

    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &crate::dtype::DType,
        _session: &VortexSession,
    ) -> VortexResult<Field> {
        let ext = dtype
            .as_extension_opt()
            .ok_or_else(|| vortex_error::vortex_err!("Expected extension dtype, got {dtype}"))?;
        let opts = ext.metadata_opt::<Timestamp>().ok_or_else(|| {
            vortex_error::vortex_err!("Expected Timestamp extension dtype, got {dtype}")
        })?;
        let arrow_unit = ArrowTimeUnit::try_from(opts.unit)?;
        Ok(Field::new(
            name,
            DataType::Timestamp(arrow_unit, opts.tz.clone()),
            dtype.is_nullable(),
        ))
    }

    fn from_arrow_field(&self, field: &Field) -> VortexResult<crate::dtype::DType> {
        let DataType::Timestamp(unit, tz) = field.data_type() else {
            vortex_bail!(
                "Timestamp plugin cannot convert Arrow type {}",
                field.data_type()
            );
        };
        Ok(crate::dtype::DType::Extension(
            Timestamp::new_with_tz(
                TimeUnit::from(*unit),
                tz.clone(),
                field.is_nullable().into(),
            )
            .erased(),
        ))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        let ext = array.dtype().as_extension_opt().ok_or_else(|| {
            vortex_error::vortex_err!("Expected Timestamp extension array, got {}", array.dtype())
        })?;
        let opts = ext.metadata_opt::<Timestamp>().ok_or_else(|| {
            vortex_error::vortex_err!("Expected Timestamp extension array, got {}", array.dtype())
        })?;

        let DataType::Timestamp(arrow_unit, arrow_tz) = target.data_type() else {
            vortex_bail!(
                "Cannot convert Timestamp array to Arrow type {}",
                target.data_type()
            );
        };
        vortex_ensure!(
            opts.tz == *arrow_tz,
            "Cannot convert {} array to Arrow type {} due to timezone mismatch",
            array.dtype(),
            target.data_type()
        );

        match (opts.unit, arrow_unit) {
            (TimeUnit::Seconds, ArrowTimeUnit::Second) => {
                to_arrow_timestamp::<TimestampSecondType>(array, arrow_tz.as_ref(), ctx)
            }
            (TimeUnit::Milliseconds, ArrowTimeUnit::Millisecond) => {
                to_arrow_timestamp::<TimestampMillisecondType>(array, arrow_tz.as_ref(), ctx)
            }
            (TimeUnit::Microseconds, ArrowTimeUnit::Microsecond) => {
                to_arrow_timestamp::<TimestampMicrosecondType>(array, arrow_tz.as_ref(), ctx)
            }
            (TimeUnit::Nanoseconds, ArrowTimeUnit::Nanosecond) => {
                to_arrow_timestamp::<TimestampNanosecondType>(array, arrow_tz.as_ref(), ctx)
            }
            (unit, arrow_unit) => vortex_bail!(
                "Cannot convert Timestamp({unit}) array to Arrow Timestamp({arrow_unit:?})"
            ),
        }
    }

    fn from_arrow_array(&self, _array: ArrowArrayRef, _field: &Field) -> VortexResult<ArrayRef> {
        vortex_bail!("Timestamp::from_arrow_array is not yet implemented")
    }
}

fn unit_for<V>(dtype: &crate::dtype::DType) -> VortexResult<TimeUnit>
where
    V: crate::dtype::extension::ExtVTable<Metadata = TimeUnit>,
{
    let ext = dtype
        .as_extension_opt()
        .ok_or_else(|| vortex_error::vortex_err!("Expected extension dtype, got {dtype}"))?;
    let unit = ext
        .metadata_opt::<V>()
        .ok_or_else(|| vortex_error::vortex_err!("Unexpected extension dtype {dtype}"))?;
    Ok(*unit)
}

fn to_temporal_array<T>(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrowArrayRef>
where
    T: ArrowTemporalType,
    T::Native: NativePType,
{
    Ok(Arc::new(to_temporal_primitive::<T>(array, ctx)?))
}

fn to_arrow_timestamp<T>(
    array: ArrayRef,
    arrow_tz: Option<&Arc<str>>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T: ArrowTimestampType,
    T::Native: NativePType,
{
    Ok(Arc::new(
        to_temporal_primitive::<T>(array, ctx)?.with_timezone_opt(arrow_tz.cloned()),
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
    let ext_array = array.execute::<ExtensionArray>(ctx)?;
    let primitive = ext_array
        .storage_array()
        .clone()
        .execute::<VortexPrimitiveArray>(ctx)?;
    vortex_ensure!(
        primitive.ptype() == T::Native::PTYPE,
        "Expected temporal storage of width {}, found {}",
        T::Native::PTYPE,
        primitive.ptype()
    );

    let validity = primitive
        .as_ref()
        .validity()?
        .execute_mask(primitive.as_ref().len(), ctx)?;
    let buffer = primitive.to_buffer::<T::Native>();

    Ok(ArrowPrimitiveArray::<T>::new(
        buffer.into_arrow_scalar_buffer(),
        to_null_buffer(validity),
    ))
}

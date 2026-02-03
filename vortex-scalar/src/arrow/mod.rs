// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Scalar as ArrowScalar;
use arrow_array::*;
use vortex_dtype::DType;
use vortex_dtype::PType;
use vortex_dtype::datetime::AnyTemporal;
use vortex_error::VortexError;
use vortex_error::vortex_bail;

use crate::Scalar;
use crate::datetime::DateValue;
use crate::datetime::TemporalValue;
use crate::datetime::TimeValue;
use crate::datetime::TimestampValue;
use crate::decimal::DecimalValue;

macro_rules! value_to_arrow_scalar {
    ($V:expr, $AR:ty) => {
        Ok(std::sync::Arc::new(
            $V.map(<$AR>::new_scalar)
                .unwrap_or_else(|| arrow_array::Scalar::new(<$AR>::new_null(1))),
        ))
    };
}

macro_rules! timestamp_to_arrow_scalar {
    ($V:expr, $TZ:expr, $AR:ty) => {{
        let array = match $V {
            Some(v) => <$AR>::new_scalar(v).into_inner(),
            None => <$AR>::new_null(1),
        }
        .with_timezone_opt($TZ);
        Ok(Arc::new(ArrowScalar::new(array)))
    }};
}

impl TryFrom<&Scalar> for Arc<dyn Datum> {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> Result<Arc<dyn Datum>, Self::Error> {
        match value.dtype() {
            DType::Null => Ok(Arc::new(NullArray::new(1))),
            DType::Bool(_) => value_to_arrow_scalar!(value.as_bool().value(), BooleanArray),
            DType::Primitive(ptype, ..) => {
                let scalar = value.as_primitive();
                Ok(match ptype {
                    PType::U8 => scalar
                        .typed_value()
                        .map(|i| Arc::new(UInt8Array::new_scalar(i)) as Arc<dyn Datum>)
                        .unwrap_or_else(|| Arc::new(UInt8Array::new_null(1))),
                    PType::U16 => scalar
                        .typed_value()
                        .map(|i| Arc::new(UInt16Array::new_scalar(i)) as Arc<dyn Datum>)
                        .unwrap_or_else(|| Arc::new(UInt16Array::new_null(1))),
                    PType::U32 => scalar
                        .typed_value()
                        .map(|i| Arc::new(UInt32Array::new_scalar(i)) as Arc<dyn Datum>)
                        .unwrap_or_else(|| Arc::new(UInt32Array::new_null(1))),
                    PType::U64 => scalar
                        .typed_value()
                        .map(|i| Arc::new(UInt64Array::new_scalar(i)) as Arc<dyn Datum>)
                        .unwrap_or_else(|| Arc::new(UInt64Array::new_null(1))),
                    PType::I8 => scalar
                        .typed_value()
                        .map(|i| Arc::new(Int8Array::new_scalar(i)) as Arc<dyn Datum>)
                        .unwrap_or_else(|| Arc::new(Int8Array::new_null(1))),
                    PType::I16 => scalar
                        .typed_value()
                        .map(|i| Arc::new(Int16Array::new_scalar(i)) as Arc<dyn Datum>)
                        .unwrap_or_else(|| Arc::new(Int16Array::new_null(1))),
                    PType::I32 => scalar
                        .typed_value()
                        .map(|i| Arc::new(Int32Array::new_scalar(i)) as Arc<dyn Datum>)
                        .unwrap_or_else(|| Arc::new(Int32Array::new_null(1))),
                    PType::I64 => scalar
                        .typed_value()
                        .map(|i| Arc::new(Int64Array::new_scalar(i)) as Arc<dyn Datum>)
                        .unwrap_or_else(|| Arc::new(Int64Array::new_null(1))),
                    PType::F16 => scalar
                        .typed_value()
                        .map(|i| Arc::new(Float16Array::new_scalar(i)) as Arc<dyn Datum>)
                        .unwrap_or_else(|| Arc::new(Float16Array::new_null(1))),
                    PType::F32 => scalar
                        .typed_value()
                        .map(|i| Arc::new(Float32Array::new_scalar(i)) as Arc<dyn Datum>)
                        .unwrap_or_else(|| Arc::new(Float32Array::new_null(1))),
                    PType::F64 => scalar
                        .typed_value()
                        .map(|i| Arc::new(Float64Array::new_scalar(i)) as Arc<dyn Datum>)
                        .unwrap_or_else(|| Arc::new(Float64Array::new_null(1))),
                })
            }
            DType::Decimal(..) => match value.as_decimal().decimal_value().cloned() {
                // TODO(joe): replace with decimal32, etc.
                Some(DecimalValue::I8(v)) => Ok(Arc::new(Decimal128Array::new_scalar(v as i128))),
                Some(DecimalValue::I16(v)) => Ok(Arc::new(Decimal128Array::new_scalar(v as i128))),
                Some(DecimalValue::I32(v)) => Ok(Arc::new(Decimal128Array::new_scalar(v as i128))),
                Some(DecimalValue::I64(v)) => Ok(Arc::new(Decimal128Array::new_scalar(v as i128))),
                Some(DecimalValue::I128(v128)) => Ok(Arc::new(Decimal128Array::new_scalar(v128))),
                Some(DecimalValue::I256(v256)) => {
                    Ok(Arc::new(Decimal256Array::new_scalar(v256.into())))
                }
                None => Ok(Arc::new(arrow_array::Scalar::new(
                    Decimal128Array::new_null(1),
                ))),
            },
            DType::Utf8(_) => {
                value_to_arrow_scalar!(value.as_utf8().value(), StringViewArray)
            }
            DType::Binary(_) => {
                value_to_arrow_scalar!(value.as_binary().value(), BinaryViewArray)
            }
            DType::Struct(..) => {
                todo!("struct scalar conversion")
            }
            DType::List(..) => {
                todo!("list scalar conversion")
            }
            DType::FixedSizeList(..) => {
                todo!("fixed-size list scalar conversion")
            }
            DType::Extension(ext) => {
                let ext_scalar = value.as_extension();
                let Some(temporal) = ext_scalar.value_opt::<AnyTemporal>() else {
                    vortex_bail!("Cannot convert extension scalar {} to Arrow", ext.id())
                };

                match temporal {
                    TemporalValue::Timestamp(TimestampValue::Nanoseconds(v, tz)) => {
                        timestamp_to_arrow_scalar!(v, tz.cloned(), TimestampNanosecondArray)
                    }
                    TemporalValue::Timestamp(TimestampValue::Microseconds(v, tz)) => {
                        timestamp_to_arrow_scalar!(v, tz.cloned(), TimestampMicrosecondArray)
                    }
                    TemporalValue::Timestamp(TimestampValue::Milliseconds(v, tz)) => {
                        timestamp_to_arrow_scalar!(v, tz.cloned(), TimestampMillisecondArray)
                    }
                    TemporalValue::Timestamp(TimestampValue::Seconds(v, tz)) => {
                        timestamp_to_arrow_scalar!(v, tz.cloned(), TimestampSecondArray)
                    }
                    TemporalValue::Date(DateValue::Days(v)) => {
                        value_to_arrow_scalar!(v, Date32Array)
                    }
                    TemporalValue::Date(DateValue::Milliseconds(v)) => {
                        value_to_arrow_scalar!(v, Date64Array)
                    }
                    TemporalValue::Time(TimeValue::Seconds(v)) => {
                        value_to_arrow_scalar!(v, Time32SecondArray)
                    }
                    TemporalValue::Time(TimeValue::Milliseconds(v)) => {
                        value_to_arrow_scalar!(v, Time32MillisecondArray)
                    }
                    TemporalValue::Time(TimeValue::Microseconds(v)) => {
                        value_to_arrow_scalar!(v, Time64MicrosecondArray)
                    }
                    TemporalValue::Time(TimeValue::Nanoseconds(v)) => {
                        value_to_arrow_scalar!(v, Time64NanosecondArray)
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;

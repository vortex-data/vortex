#![cfg(feature = "datafusion")]
use std::sync::Arc;

use datafusion_common::ScalarValue;
use vortex_buffer::Buffer;
use vortex_datetime_dtype::arrow::make_temporal_ext_dtype;
use vortex_datetime_dtype::{is_temporal_ext_type, TemporalMetadata, TimeUnit};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::VortexError;

use crate::{PValue, Scalar};

impl TryFrom<Scalar> for ScalarValue {
    type Error = VortexError;

    fn try_from(value: Scalar) -> Result<Self, Self::Error> {
        let (dtype, value) = value.into_parts();
        Ok(match dtype {
            DType::Null => ScalarValue::Null,
            DType::Bool(_) => ScalarValue::Boolean(value.as_bool()?),
            DType::Primitive(ptype, _) => {
                let pvalue = value.as_pvalue()?;
                match pvalue {
                    None => match ptype {
                        PType::U8 => ScalarValue::UInt8(None),
                        PType::U16 => ScalarValue::UInt16(None),
                        PType::U32 => ScalarValue::UInt32(None),
                        PType::U64 => ScalarValue::UInt64(None),
                        PType::I8 => ScalarValue::Int8(None),
                        PType::I16 => ScalarValue::Int16(None),
                        PType::I32 => ScalarValue::Int32(None),
                        PType::I64 => ScalarValue::Int64(None),
                        PType::F16 => ScalarValue::Float16(None),
                        PType::F32 => ScalarValue::Float32(None),
                        PType::F64 => ScalarValue::Float64(None),
                    },
                    Some(pvalue) => match pvalue {
                        PValue::U8(v) => ScalarValue::UInt8(Some(v)),
                        PValue::U16(v) => ScalarValue::UInt16(Some(v)),
                        PValue::U32(v) => ScalarValue::UInt32(Some(v)),
                        PValue::U64(v) => ScalarValue::UInt64(Some(v)),
                        PValue::I8(v) => ScalarValue::Int8(Some(v)),
                        PValue::I16(v) => ScalarValue::Int16(Some(v)),
                        PValue::I32(v) => ScalarValue::Int32(Some(v)),
                        PValue::I64(v) => ScalarValue::Int64(Some(v)),
                        PValue::F16(v) => ScalarValue::Float16(Some(v)),
                        PValue::F32(v) => ScalarValue::Float32(Some(v)),
                        PValue::F64(v) => ScalarValue::Float64(Some(v)),
                    },
                }
            }
            DType::Utf8(_) => {
                ScalarValue::Utf8(value.as_buffer_string()?.map(|b| b.as_str().to_string()))
            }
            DType::Binary(_) => ScalarValue::Binary(
                value
                    .as_buffer()?
                    .map(|b| b.into_vec().unwrap_or_else(|buf| buf.as_slice().to_vec())),
            ),
            DType::Struct(..) => {
                todo!("struct scalar conversion")
            }
            DType::List(..) => {
                todo!("list scalar conversion")
            }
            DType::Extension(ext) => {
                // Special handling: temporal extension types in Vortex correspond to Arrow's
                // temporal physical types.
                if is_temporal_ext_type(ext.id()) {
                    let metadata = TemporalMetadata::try_from(ext.as_ref())?;
                    let pv = value.as_pvalue()?;
                    return Ok(match metadata {
                        TemporalMetadata::Time(u) => match u {
                            TimeUnit::Ns => {
                                ScalarValue::Time64Nanosecond(pv.and_then(|p| p.as_i64()))
                            }
                            TimeUnit::Us => {
                                ScalarValue::Time64Microsecond(pv.and_then(|p| p.as_i64()))
                            }
                            TimeUnit::Ms => {
                                ScalarValue::Time32Millisecond(pv.and_then(|p| p.as_i32()))
                            }
                            TimeUnit::S => ScalarValue::Time32Second(pv.and_then(|p| p.as_i32())),
                            TimeUnit::D => {
                                unreachable!("Unsupported TimeUnit {u} for {}", ext.id())
                            }
                        },
                        TemporalMetadata::Date(u) => match u {
                            TimeUnit::Ms => ScalarValue::Date64(pv.and_then(|p| p.as_i64())),
                            TimeUnit::D => ScalarValue::Date32(pv.and_then(|p| p.as_i32())),
                            _ => unreachable!("Unsupported TimeUnit {u} for {}", ext.id()),
                        },
                        TemporalMetadata::Timestamp(u, tz) => match u {
                            TimeUnit::Ns => ScalarValue::TimestampNanosecond(
                                pv.and_then(|p| p.as_i64()),
                                tz.map(|t| t.into()),
                            ),
                            TimeUnit::Us => ScalarValue::TimestampMicrosecond(
                                pv.and_then(|p| p.as_i64()),
                                tz.map(|t| t.into()),
                            ),
                            TimeUnit::Ms => ScalarValue::TimestampMillisecond(
                                pv.and_then(|p| p.as_i64()),
                                tz.map(|t| t.into()),
                            ),
                            TimeUnit::S => ScalarValue::TimestampSecond(
                                pv.and_then(|p| p.as_i64()),
                                tz.map(|t| t.into()),
                            ),
                            TimeUnit::D => {
                                unreachable!("Unsupported TimeUnit {u} for {}", ext.id())
                            }
                        },
                    });
                } else {
                    // Unknown extension type: perform scalar conversion using the canonical
                    // scalar DType.
                    ScalarValue::try_from(Scalar::new(ext.storage_dtype().clone(), value))?
                }
            }
        })
    }
}

impl From<ScalarValue> for Scalar {
    fn from(value: ScalarValue) -> Scalar {
        match value {
            ScalarValue::Null => Some(Scalar::null(DType::Null)),
            ScalarValue::Boolean(b) => b.map(Scalar::from),
            ScalarValue::Float16(f) => f.map(Scalar::from),
            ScalarValue::Float32(f) => f.map(Scalar::from),
            ScalarValue::Float64(f) => f.map(Scalar::from),
            ScalarValue::Int8(i) => i.map(Scalar::from),
            ScalarValue::Int16(i) => i.map(Scalar::from),
            ScalarValue::Int32(i) => i.map(Scalar::from),
            ScalarValue::Int64(i) => i.map(Scalar::from),
            ScalarValue::UInt8(i) => i.map(Scalar::from),
            ScalarValue::UInt16(i) => i.map(Scalar::from),
            ScalarValue::UInt32(i) => i.map(Scalar::from),
            ScalarValue::UInt64(i) => i.map(Scalar::from),
            ScalarValue::Utf8(s) | ScalarValue::Utf8View(s) | ScalarValue::LargeUtf8(s) => {
                s.as_ref().map(|s| Scalar::from(s.as_str()))
            }
            ScalarValue::Binary(b)
            | ScalarValue::BinaryView(b)
            | ScalarValue::LargeBinary(b)
            | ScalarValue::FixedSizeBinary(_, b) => b
                .as_ref()
                .map(|b| Scalar::binary(Buffer::from(b.clone()), Nullability::Nullable)),
            ScalarValue::Date32(v)
            | ScalarValue::Time32Second(v)
            | ScalarValue::Time32Millisecond(v) => v.map(|i| {
                let ext_dtype = make_temporal_ext_dtype(&value.data_type())
                    .with_nullability(Nullability::Nullable);
                Scalar::new(
                    DType::Extension(Arc::new(ext_dtype)),
                    crate::ScalarValue::Primitive(PValue::I32(i)),
                )
            }),
            ScalarValue::Date64(v)
            | ScalarValue::Time64Microsecond(v)
            | ScalarValue::Time64Nanosecond(v)
            | ScalarValue::TimestampSecond(v, _)
            | ScalarValue::TimestampMillisecond(v, _)
            | ScalarValue::TimestampMicrosecond(v, _)
            | ScalarValue::TimestampNanosecond(v, _) => v.map(|i| {
                let ext_dtype = make_temporal_ext_dtype(&value.data_type());
                Scalar::new(
                    DType::Extension(Arc::new(ext_dtype.with_nullability(Nullability::Nullable))),
                    crate::ScalarValue::Primitive(PValue::I64(i)),
                )
            }),
            _ => unimplemented!("Can't convert {value:?} value to a Vortex scalar"),
        }
        .unwrap_or_else(|| Scalar::null(DType::Null))
    }
}

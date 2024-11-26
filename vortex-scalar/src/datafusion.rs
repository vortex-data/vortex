#![cfg(feature = "datafusion")]
use std::sync::Arc;

use datafusion_common::ScalarValue;
use vortex_buffer::Buffer;
use vortex_datetime_dtype::arrow::make_temporal_ext_dtype;
use vortex_datetime_dtype::{is_temporal_ext_type, TemporalMetadata, TimeUnit};
use vortex_dtype::half::f16;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::VortexError;

use crate::{InnerScalarValue, PValue, Scalar};

impl TryFrom<Scalar> for ScalarValue {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        Ok(match scalar.dtype() {
            DType::Null => ScalarValue::Null,
            DType::Bool(_) => ScalarValue::Boolean(scalar.as_bool().value()),
            DType::Primitive(ptype, _) => {
                let pscalar = scalar.as_primitive();
                match ptype {
                    PType::U8 => ScalarValue::UInt8(pscalar.typed_value::<u8>()),
                    PType::U16 => ScalarValue::UInt16(pscalar.typed_value::<u16>()),
                    PType::U32 => ScalarValue::UInt32(pscalar.typed_value::<u32>()),
                    PType::U64 => ScalarValue::UInt64(pscalar.typed_value::<u64>()),
                    PType::I8 => ScalarValue::Int8(pscalar.typed_value::<i8>()),
                    PType::I16 => ScalarValue::Int16(pscalar.typed_value::<i16>()),
                    PType::I32 => ScalarValue::Int32(pscalar.typed_value::<i32>()),
                    PType::I64 => ScalarValue::Int64(pscalar.typed_value::<i64>()),
                    PType::F16 => ScalarValue::Float16(pscalar.typed_value::<f16>()),
                    PType::F32 => ScalarValue::Float32(pscalar.typed_value::<f32>()),
                    PType::F64 => ScalarValue::Float64(pscalar.typed_value::<f64>()),
                }
            }
            DType::Utf8(_) => {
                ScalarValue::Utf8(scalar.as_utf8().value().map(|s| s.as_str().to_string()))
            }
            DType::Binary(_) => ScalarValue::Binary(
                scalar
                    .as_binary()
                    .value()
                    .map(|b| b.into_vec().unwrap_or_else(|buf| buf.as_slice().to_vec())),
            ),
            DType::Struct(..) => {
                todo!("struct scalar conversion")
            }
            DType::List(..) => {
                todo!("list scalar conversion")
            }
            DType::Extension(ext) => {
                let storage_scalar = scalar.as_extension().storage();

                // Special handling: temporal extension types in Vortex correspond to Arrow's
                // temporal physical types.
                if is_temporal_ext_type(ext.id()) {
                    let metadata = TemporalMetadata::try_from(ext.as_ref())?;
                    let pv = storage_scalar.as_primitive();
                    return Ok(match metadata {
                        TemporalMetadata::Time(u) => match u {
                            TimeUnit::Ns => ScalarValue::Time64Nanosecond(pv.as_::<i64>()?),
                            TimeUnit::Us => ScalarValue::Time64Microsecond(pv.as_::<i64>()?),
                            TimeUnit::Ms => ScalarValue::Time32Millisecond(pv.as_::<i32>()?),
                            TimeUnit::S => ScalarValue::Time32Second(pv.as_::<i32>()?),
                            TimeUnit::D => {
                                unreachable!("Unsupported TimeUnit {u} for {}", ext.id())
                            }
                        },
                        TemporalMetadata::Date(u) => match u {
                            TimeUnit::Ms => ScalarValue::Date64(pv.as_::<i64>()?),
                            TimeUnit::D => ScalarValue::Date32(pv.as_::<i32>()?),
                            _ => unreachable!("Unsupported TimeUnit {u} for {}", ext.id()),
                        },
                        TemporalMetadata::Timestamp(u, tz) => match u {
                            TimeUnit::Ns => ScalarValue::TimestampNanosecond(
                                pv.as_::<i64>()?,
                                tz.map(|t| t.into()),
                            ),
                            TimeUnit::Us => ScalarValue::TimestampMicrosecond(
                                pv.as_::<i64>()?,
                                tz.map(|t| t.into()),
                            ),
                            TimeUnit::Ms => ScalarValue::TimestampMillisecond(
                                pv.as_::<i64>()?,
                                tz.map(|t| t.into()),
                            ),
                            TimeUnit::S => {
                                ScalarValue::TimestampSecond(pv.as_::<i64>()?, tz.map(|t| t.into()))
                            }
                            TimeUnit::D => {
                                unreachable!("Unsupported TimeUnit {u} for {}", ext.id())
                            }
                        },
                    });
                } else {
                    // Unknown extension type: perform scalar conversion using the canonical
                    // scalar DType.
                    ScalarValue::try_from(storage_scalar)?
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
                    crate::ScalarValue(InnerScalarValue::Primitive(PValue::I32(i))),
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
                    crate::ScalarValue(InnerScalarValue::Primitive(PValue::I64(i))),
                )
            }),
            _ => unimplemented!("Can't convert {value:?} value to a Vortex scalar"),
        }
        .unwrap_or_else(|| Scalar::null(DType::Null))
    }
}

#![cfg(feature = "datafusion")]

use std::sync::Arc;

use datafusion_common::ScalarValue as DFScalarValue;
use vortex_buffer::ByteBuffer;
use vortex_datetime_dtype::arrow::make_temporal_ext_dtype;
use vortex_datetime_dtype::{is_temporal_ext_type, TemporalMetadata, TimeUnit};
use vortex_dtype::dtypes::DTYPE_NULL;
use vortex_dtype::half::f16;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::VortexError;

use crate::{InnerScalarValue, PValue, Scalar};

impl TryFrom<Scalar> for DFScalarValue {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        Ok(match scalar.dtype().as_ref() {
            DType::Null => DFScalarValue::Null,
            DType::Bool(_) => DFScalarValue::Boolean(scalar.as_bool().value()),
            DType::Primitive(ptype, _) => {
                let pscalar = scalar.as_primitive();
                match ptype {
                    PType::U8 => DFScalarValue::UInt8(pscalar.typed_value::<u8>()),
                    PType::U16 => DFScalarValue::UInt16(pscalar.typed_value::<u16>()),
                    PType::U32 => DFScalarValue::UInt32(pscalar.typed_value::<u32>()),
                    PType::U64 => DFScalarValue::UInt64(pscalar.typed_value::<u64>()),
                    PType::I8 => DFScalarValue::Int8(pscalar.typed_value::<i8>()),
                    PType::I16 => DFScalarValue::Int16(pscalar.typed_value::<i16>()),
                    PType::I32 => DFScalarValue::Int32(pscalar.typed_value::<i32>()),
                    PType::I64 => DFScalarValue::Int64(pscalar.typed_value::<i64>()),
                    PType::F16 => DFScalarValue::Float16(pscalar.typed_value::<f16>()),
                    PType::F32 => DFScalarValue::Float32(pscalar.typed_value::<f32>()),
                    PType::F64 => DFScalarValue::Float64(pscalar.typed_value::<f64>()),
                }
            }
            DType::Utf8(_) => {
                DFScalarValue::Utf8(scalar.as_utf8().value().map(|s| s.as_str().to_string()))
            }
            DType::Binary(_) => {
                DFScalarValue::Binary(scalar.as_binary().value().map(|b| b.as_slice().to_vec()))
            }
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
                            TimeUnit::Ns => DFScalarValue::Time64Nanosecond(pv.as_::<i64>()?),
                            TimeUnit::Us => DFScalarValue::Time64Microsecond(pv.as_::<i64>()?),
                            TimeUnit::Ms => DFScalarValue::Time32Millisecond(pv.as_::<i32>()?),
                            TimeUnit::S => DFScalarValue::Time32Second(pv.as_::<i32>()?),
                            TimeUnit::D => {
                                unreachable!("Unsupported TimeUnit {u} for {}", ext.id())
                            }
                        },
                        TemporalMetadata::Date(u) => match u {
                            TimeUnit::Ms => DFScalarValue::Date64(pv.as_::<i64>()?),
                            TimeUnit::D => DFScalarValue::Date32(pv.as_::<i32>()?),
                            _ => unreachable!("Unsupported TimeUnit {u} for {}", ext.id()),
                        },
                        TemporalMetadata::Timestamp(u, tz) => match u {
                            TimeUnit::Ns => DFScalarValue::TimestampNanosecond(
                                pv.as_::<i64>()?,
                                tz.map(|t| t.into()),
                            ),
                            TimeUnit::Us => DFScalarValue::TimestampMicrosecond(
                                pv.as_::<i64>()?,
                                tz.map(|t| t.into()),
                            ),
                            TimeUnit::Ms => DFScalarValue::TimestampMillisecond(
                                pv.as_::<i64>()?,
                                tz.map(|t| t.into()),
                            ),
                            TimeUnit::S => DFScalarValue::TimestampSecond(
                                pv.as_::<i64>()?,
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
                    DFScalarValue::try_from(storage_scalar)?
                }
            }
        })
    }
}

impl From<DFScalarValue> for Scalar {
    fn from(value: DFScalarValue) -> Scalar {
        match value {
            DFScalarValue::Null => None,
            DFScalarValue::Boolean(b) => b.map(Scalar::from),
            DFScalarValue::Float16(f) => f.map(Scalar::from),
            DFScalarValue::Float32(f) => f.map(Scalar::from),
            DFScalarValue::Float64(f) => f.map(Scalar::from),
            DFScalarValue::Int8(i) => i.map(Scalar::from),
            DFScalarValue::Int16(i) => i.map(Scalar::from),
            DFScalarValue::Int32(i) => i.map(Scalar::from),
            DFScalarValue::Int64(i) => i.map(Scalar::from),
            DFScalarValue::UInt8(i) => i.map(Scalar::from),
            DFScalarValue::UInt16(i) => i.map(Scalar::from),
            DFScalarValue::UInt32(i) => i.map(Scalar::from),
            DFScalarValue::UInt64(i) => i.map(Scalar::from),
            DFScalarValue::Utf8(s) | DFScalarValue::Utf8View(s) | DFScalarValue::LargeUtf8(s) => {
                s.as_ref().map(|s| Scalar::from(s.as_str()))
            }
            DFScalarValue::Binary(b)
            | DFScalarValue::BinaryView(b)
            | DFScalarValue::LargeBinary(b)
            | DFScalarValue::FixedSizeBinary(_, b) => b
                .as_ref()
                .map(|b| Scalar::binary(ByteBuffer::from(b.clone()), Nullability::Nullable)),
            DFScalarValue::Date32(v)
            | DFScalarValue::Time32Second(v)
            | DFScalarValue::Time32Millisecond(v) => v.map(|i| {
                let ext_dtype = make_temporal_ext_dtype(&value.data_type())
                    .with_nullability(Nullability::Nullable);
                Scalar::new(
                    Arc::new(DType::Extension(Arc::new(ext_dtype))),
                    crate::ScalarValue(InnerScalarValue::Primitive(PValue::I32(i))),
                )
            }),
            DFScalarValue::Date64(v)
            | DFScalarValue::Time64Microsecond(v)
            | DFScalarValue::Time64Nanosecond(v)
            | DFScalarValue::TimestampSecond(v, _)
            | DFScalarValue::TimestampMillisecond(v, _)
            | DFScalarValue::TimestampMicrosecond(v, _)
            | DFScalarValue::TimestampNanosecond(v, _) => v.map(|i| {
                let ext_dtype = make_temporal_ext_dtype(&value.data_type());
                Scalar::new(
                    Arc::new(DType::Extension(Arc::new(
                        ext_dtype.with_nullability(Nullability::Nullable),
                    ))),
                    crate::ScalarValue(InnerScalarValue::Primitive(PValue::I64(i))),
                )
            }),
            _ => unimplemented!("Can't convert {value:?} value to a Vortex scalar"),
        }
        .unwrap_or_else(|| Scalar::null(DTYPE_NULL.clone()))
    }
}

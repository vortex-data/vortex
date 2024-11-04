use std::sync::Arc;

use arrow_array::*;
use vortex_datetime_dtype::{is_temporal_ext_type, TemporalMetadata, TimeUnit};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexError};

use crate::{PValue, Scalar};

macro_rules! value_to_arrow_scalar {
    ($V:expr, $AR:ty) => {
        Ok(std::sync::Arc::new(
            $V.map(<$AR>::new_scalar)
                .unwrap_or_else(|| arrow_array::Scalar::new(<$AR>::new_null(1))),
        ))
    };
}

impl TryFrom<&Scalar> for Arc<dyn Datum> {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> Result<Arc<dyn Datum>, Self::Error> {
        match value.dtype() {
            DType::Null => Ok(Arc::new(NullArray::new(1))),
            DType::Bool(_) => value_to_arrow_scalar!(value.value.as_bool()?, BooleanArray),
            DType::Primitive(ptype, _) => {
                let pvalue = value.value.as_pvalue()?;
                Ok(match pvalue {
                    None => match ptype {
                        PType::U8 => Arc::new(UInt8Array::new_null(1)),
                        PType::U16 => Arc::new(UInt16Array::new_null(1)),
                        PType::U32 => Arc::new(UInt32Array::new_null(1)),
                        PType::U64 => Arc::new(UInt64Array::new_null(1)),
                        PType::I8 => Arc::new(Int8Array::new_null(1)),
                        PType::I16 => Arc::new(Int16Array::new_null(1)),
                        PType::I32 => Arc::new(Int32Array::new_null(1)),
                        PType::I64 => Arc::new(Int64Array::new_null(1)),
                        PType::F16 => Arc::new(Float16Array::new_null(1)),
                        PType::F32 => Arc::new(Float32Array::new_null(1)),
                        PType::F64 => Arc::new(Float64Array::new_null(1)),
                    },
                    Some(pvalue) => match pvalue {
                        PValue::U8(v) => Arc::new(UInt8Array::new_scalar(v)),
                        PValue::U16(v) => Arc::new(UInt16Array::new_scalar(v)),
                        PValue::U32(v) => Arc::new(UInt32Array::new_scalar(v)),
                        PValue::U64(v) => Arc::new(UInt64Array::new_scalar(v)),
                        PValue::I8(v) => Arc::new(Int8Array::new_scalar(v)),
                        PValue::I16(v) => Arc::new(Int16Array::new_scalar(v)),
                        PValue::I32(v) => Arc::new(Int32Array::new_scalar(v)),
                        PValue::I64(v) => Arc::new(Int64Array::new_scalar(v)),
                        PValue::F16(v) => Arc::new(Float16Array::new_scalar(v)),
                        PValue::F32(v) => Arc::new(Float32Array::new_scalar(v)),
                        PValue::F64(v) => Arc::new(Float64Array::new_scalar(v)),
                    },
                })
            }
            DType::Utf8(_) => {
                value_to_arrow_scalar!(value.value.as_buffer_string()?, StringViewArray)
            }
            DType::Binary(_) => {
                value_to_arrow_scalar!(value.value.as_buffer()?, BinaryViewArray)
            }
            DType::Struct(..) => {
                todo!("struct scalar conversion")
            }
            DType::List(..) => {
                todo!("list scalar conversion")
            }
            DType::Extension(ext) => {
                if is_temporal_ext_type(ext.id()) {
                    let metadata = TemporalMetadata::try_from(ext.as_ref())?;
                    let pv = value.value.as_pvalue()?;
                    return match metadata {
                        TemporalMetadata::Time(u) => match u {
                            TimeUnit::Ns => value_to_arrow_scalar!(
                                pv.and_then(|p| p.as_i64()),
                                Time64NanosecondArray
                            ),
                            TimeUnit::Us => value_to_arrow_scalar!(
                                pv.and_then(|p| p.as_i64()),
                                Time64MicrosecondArray
                            ),
                            TimeUnit::Ms => value_to_arrow_scalar!(
                                pv.and_then(|p| p.as_i32()),
                                Time32MillisecondArray
                            ),
                            TimeUnit::S => value_to_arrow_scalar!(
                                pv.and_then(|p| p.as_i32()),
                                Time32SecondArray
                            ),
                            TimeUnit::D => {
                                vortex_bail!("Unsupported TimeUnit {u} for {}", ext.id())
                            }
                        },
                        TemporalMetadata::Date(u) => match u {
                            TimeUnit::Ms => {
                                value_to_arrow_scalar!(pv.and_then(|p| p.as_i64()), Date64Array)
                            }
                            TimeUnit::D => {
                                value_to_arrow_scalar!(pv.and_then(|p| p.as_i32()), Date32Array)
                            }
                            _ => vortex_bail!("Unsupported TimeUnit {u} for {}", ext.id()),
                        },
                        TemporalMetadata::Timestamp(u, _) => match u {
                            TimeUnit::Ns => value_to_arrow_scalar!(
                                pv.and_then(|p| p.as_i64()),
                                TimestampNanosecondArray
                            ),
                            TimeUnit::Us => value_to_arrow_scalar!(
                                pv.and_then(|p| p.as_i64()),
                                TimestampMicrosecondArray
                            ),
                            TimeUnit::Ms => value_to_arrow_scalar!(
                                pv.and_then(|p| p.as_i64()),
                                TimestampMillisecondArray
                            ),
                            TimeUnit::S => value_to_arrow_scalar!(
                                pv.and_then(|p| p.as_i64()),
                                TimestampSecondArray
                            ),
                            TimeUnit::D => {
                                vortex_bail!("Unsupported TimeUnit {u} for {}", ext.id())
                            }
                        },
                    };
                }

                todo!("Non temporal extension scalar conversion")
            }
        }
    }
}

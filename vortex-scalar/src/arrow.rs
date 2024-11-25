use std::sync::Arc;

use arrow_array::*;
use vortex_datetime_dtype::{is_temporal_ext_type, TemporalMetadata, TimeUnit};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, vortex_err, VortexError};

use crate::Scalar;

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
            DType::Extension(ext) => {
                if is_temporal_ext_type(ext.id()) {
                    let metadata = TemporalMetadata::try_from(ext.as_ref())?;
                    let storage_scalar = value.as_extension().storage();
                    let primitive = storage_scalar
                        .as_primitive_opt()
                        .ok_or_else(|| vortex_err!("Expected primitive scalar"))?;

                    return match metadata {
                        TemporalMetadata::Time(u) => match u {
                            TimeUnit::Ns => value_to_arrow_scalar!(
                                primitive.as_::<i64>()?,
                                Time64NanosecondArray
                            ),
                            TimeUnit::Us => value_to_arrow_scalar!(
                                primitive.as_::<i64>()?,
                                Time64MicrosecondArray
                            ),
                            TimeUnit::Ms => value_to_arrow_scalar!(
                                primitive.as_::<i32>()?,
                                Time32MillisecondArray
                            ),
                            TimeUnit::S => {
                                value_to_arrow_scalar!(primitive.as_::<i32>()?, Time32SecondArray)
                            }
                            TimeUnit::D => {
                                vortex_bail!("Unsupported TimeUnit {u} for {}", ext.id())
                            }
                        },
                        TemporalMetadata::Date(u) => match u {
                            TimeUnit::Ms => {
                                value_to_arrow_scalar!(primitive.as_::<i64>()?, Date64Array)
                            }
                            TimeUnit::D => {
                                value_to_arrow_scalar!(primitive.as_::<i32>()?, Date32Array)
                            }
                            _ => vortex_bail!("Unsupported TimeUnit {u} for {}", ext.id()),
                        },
                        TemporalMetadata::Timestamp(u, _) => match u {
                            TimeUnit::Ns => value_to_arrow_scalar!(
                                primitive.as_::<i64>()?,
                                TimestampNanosecondArray
                            ),
                            TimeUnit::Us => value_to_arrow_scalar!(
                                primitive.as_::<i64>()?,
                                TimestampMicrosecondArray
                            ),
                            TimeUnit::Ms => value_to_arrow_scalar!(
                                primitive.as_::<i64>()?,
                                TimestampMillisecondArray
                            ),
                            TimeUnit::S => value_to_arrow_scalar!(
                                primitive.as_::<i64>()?,
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

use vortex_buffer::{Buffer, BufferString};
use vortex_dtype::half::f16;
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};
use vortex_proto::scalar as pb;
use vortex_proto::scalar::scalar_value::Kind;
use vortex_proto::scalar::ListValue;

use crate::pvalue::PValue;
use crate::{InnerScalarValue, Scalar, ScalarValue};

impl From<&Scalar> for pb::Scalar {
    fn from(value: &Scalar) -> Self {
        pb::Scalar {
            dtype: Some((&value.dtype).into()),
            value: Some((&value.value).into()),
        }
    }
}

impl From<&ScalarValue> for pb::ScalarValue {
    fn from(value: &ScalarValue) -> Self {
        match value {
            ScalarValue(InnerScalarValue::Null) => pb::ScalarValue {
                kind: Some(Kind::NullValue(0)),
            },
            ScalarValue(InnerScalarValue::Bool(v)) => pb::ScalarValue {
                kind: Some(Kind::BoolValue(*v)),
            },
            ScalarValue(InnerScalarValue::Primitive(v)) => v.into(),
            ScalarValue(InnerScalarValue::Buffer(v)) => pb::ScalarValue {
                kind: Some(Kind::BytesValue(v.as_slice().to_vec())),
            },
            ScalarValue(InnerScalarValue::BufferString(v)) => pb::ScalarValue {
                kind: Some(Kind::StringValue(v.as_str().to_string())),
            },
            ScalarValue(InnerScalarValue::List(v)) => {
                let mut values = Vec::with_capacity(v.len());
                for elem in v.iter() {
                    // FIXME(DK): protobufs should probably also have inner vs non-inner?
                    values.push(pb::ScalarValue::from(&ScalarValue(elem.clone())));
                }
                pb::ScalarValue {
                    kind: Some(Kind::ListValue(ListValue { values })),
                }
            }
        }
    }
}

impl From<&PValue> for pb::ScalarValue {
    fn from(value: &PValue) -> Self {
        match value {
            PValue::I8(v) => pb::ScalarValue {
                kind: Some(Kind::Int32Value(*v as i32)),
            },
            PValue::I16(v) => pb::ScalarValue {
                kind: Some(Kind::Int32Value(*v as i32)),
            },
            PValue::I32(v) => pb::ScalarValue {
                kind: Some(Kind::Int32Value(*v)),
            },
            PValue::I64(v) => pb::ScalarValue {
                kind: Some(Kind::Int64Value(*v)),
            },
            PValue::U8(v) => pb::ScalarValue {
                kind: Some(Kind::Uint32Value(*v as u32)),
            },
            PValue::U16(v) => pb::ScalarValue {
                kind: Some(Kind::Uint32Value(*v as u32)),
            },
            PValue::U32(v) => pb::ScalarValue {
                kind: Some(Kind::Uint32Value(*v)),
            },
            PValue::U64(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(*v)),
            },
            PValue::F16(v) => pb::ScalarValue {
                kind: Some(Kind::Uint32Value(v.to_bits() as u32)),
            },
            PValue::F32(v) => pb::ScalarValue {
                kind: Some(Kind::FloatValue(*v)),
            },
            PValue::F64(v) => pb::ScalarValue {
                kind: Some(Kind::DoubleValue(*v)),
            },
        }
    }
}

impl TryFrom<&pb::Scalar> for Scalar {
    type Error = VortexError;

    fn try_from(value: &pb::Scalar) -> Result<Self, Self::Error> {
        let dtype = DType::try_from(
            value
                .dtype
                .as_ref()
                .ok_or_else(|| vortex_err!(InvalidSerde: "Scalar missing dtype"))?,
        )?;

        let value = deserialize_scalar_value(
            &dtype,
            value
                .value
                .as_ref()
                .ok_or_else(|| vortex_err!(InvalidSerde: "Scalar missing value"))?,
        )?;

        Ok(Self { dtype, value })
    }
}

fn deserialize_scalar_value(dtype: &DType, value: &pb::ScalarValue) -> VortexResult<ScalarValue> {
    let kind = value
        .kind
        .as_ref()
        .ok_or_else(|| vortex_err!(InvalidSerde: "ScalarValue missing kind"))?;

    match kind {
        Kind::NullValue(_) => Ok(ScalarValue(InnerScalarValue::Null)),
        Kind::BoolValue(v) => Ok(ScalarValue(InnerScalarValue::Bool(*v))),
        Kind::Int32Value(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::I32(*v)))),
        Kind::Int64Value(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::I64(*v)))),
        Kind::Uint32Value(v) => match dtype {
            DType::Primitive(PType::F16, _) => {
                let f16_value = f16::from_bits(u16::try_from(*v).map_err(|_| {
                    vortex_err!(
                        "expected f16 encoded as u16 inside u32 but found too large value {:#x}",
                        v
                    )
                })?);

                Ok(ScalarValue(InnerScalarValue::Primitive(PValue::F16(
                    f16_value,
                ))))
            }
            DType::Primitive(PType::U32, _) => {
                Ok(ScalarValue(InnerScalarValue::Primitive(PValue::U32(*v))))
            }
            _ => vortex_bail!("invalid dtype for f32 value {}", dtype),
        },
        Kind::Uint64Value(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::U64(*v)))),
        Kind::FloatValue(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::F32(*v)))),
        Kind::DoubleValue(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::F64(*v)))),
        Kind::StringValue(v) => Ok(ScalarValue(InnerScalarValue::BufferString(
            BufferString::from(v.clone()),
        ))),
        Kind::BytesValue(v) => Ok(ScalarValue(InnerScalarValue::Buffer(Buffer::from(
            v.as_slice(),
        )))),
        Kind::ListValue(v) => {
            let mut values = Vec::with_capacity(v.values.len());
            match dtype {
                DType::Struct(structdt, _) => {
                    for (elem, dtype) in v.values.iter().zip(structdt.dtypes().iter()) {
                        values.push(deserialize_scalar_value(dtype, elem)?);
                    }
                }
                DType::List(elementdt, _) => {
                    for elem in v.values.iter() {
                        values.push(deserialize_scalar_value(elementdt, elem)?);
                    }
                }
                _ => vortex_bail!("invalid dtype for list value {}", dtype),
            }
            Ok(ScalarValue(InnerScalarValue::List(
                values.iter().map(|x| x.0.clone()).collect(),
            )))
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use vortex_buffer::BufferString;
    use vortex_dtype::half::f16;
    use vortex_dtype::PType::{self, I32};
    use vortex_dtype::{DType, Nullability};
    use vortex_proto::scalar as pb;

    use crate::{InnerScalarValue, PValue, Scalar, ScalarValue};

    fn round_trip(scalar: Scalar) {
        assert_eq!(
            scalar,
            Scalar::try_from(&pb::Scalar::from(&scalar)).unwrap(),
        );
    }

    #[test]
    fn test_null() {
        round_trip(Scalar::null(DType::Null));
    }

    #[test]
    fn test_bool() {
        round_trip(Scalar::new(
            DType::Bool(Nullability::Nullable),
            ScalarValue(InnerScalarValue::Bool(true)),
        ));
    }

    #[test]
    fn test_primitive() {
        round_trip(Scalar::new(
            DType::Primitive(I32, Nullability::Nullable),
            ScalarValue(InnerScalarValue::Primitive(42i32.into())),
        ));
    }

    #[test]
    fn test_buffer() {
        round_trip(Scalar::new(
            DType::Binary(Nullability::Nullable),
            ScalarValue(InnerScalarValue::Buffer(vec![1, 2, 3].into())),
        ));
    }

    #[test]
    fn test_buffer_string() {
        round_trip(Scalar::new(
            DType::Utf8(Nullability::Nullable),
            ScalarValue(InnerScalarValue::BufferString(BufferString::from(
                "hello".to_string(),
            ))),
        ));
    }

    #[test]
    fn test_list() {
        round_trip(Scalar::new(
            DType::List(
                Arc::new(DType::Primitive(I32, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            ScalarValue(InnerScalarValue::List(
                vec![
                    InnerScalarValue::Primitive(42i32.into()),
                    InnerScalarValue::Primitive(43i32.into()),
                ]
                .into(),
            )),
        ));
    }

    #[test]
    fn test_f16() {
        round_trip(Scalar::new(
            DType::Primitive(PType::F16, Nullability::Nullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::F16(f16::from_f32(
                0.42,
            )))),
        ));
    }
}

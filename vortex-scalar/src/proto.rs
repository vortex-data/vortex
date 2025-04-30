use std::sync::Arc;

use num_traits::ToBytes;
use vortex_buffer::{BufferString, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexError, vortex_err};
use vortex_proto::scalar as pb;
use vortex_proto::scalar::ListValue;
use vortex_proto::scalar::scalar_value::Kind;

use crate::pvalue::PValue;
use crate::{DecimalValue, InnerScalarValue, Scalar, ScalarValue};

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
            ScalarValue(InnerScalarValue::Decimal(v)) => {
                let inner_value = match v {
                    DecimalValue::I8(v) => v.to_le_bytes().to_vec(),
                    DecimalValue::I16(v) => v.to_le_bytes().to_vec(),
                    DecimalValue::I32(v) => v.to_le_bytes().to_vec(),
                    DecimalValue::I64(v) => v.to_le_bytes().to_vec(),
                    DecimalValue::I128(v128) => v128.to_le_bytes().to_vec(),
                    DecimalValue::I256(v256) => v256.to_le_bytes().to_vec(),
                };

                pb::ScalarValue {
                    kind: Some(Kind::BytesValue(inner_value)),
                }
            }
            ScalarValue(InnerScalarValue::Buffer(v)) => pb::ScalarValue {
                kind: Some(Kind::BytesValue(v.as_slice().to_vec())),
            },
            ScalarValue(InnerScalarValue::BufferString(v)) => pb::ScalarValue {
                kind: Some(Kind::StringValue(v.as_str().to_string())),
            },
            ScalarValue(InnerScalarValue::List(v)) => {
                let mut values = Vec::with_capacity(v.len());
                for elem in v.iter() {
                    values.push(pb::ScalarValue::from(elem));
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
                kind: Some(Kind::Int64Value(*v as i64)),
            },
            PValue::I16(v) => pb::ScalarValue {
                kind: Some(Kind::Int64Value(*v as i64)),
            },
            PValue::I32(v) => pb::ScalarValue {
                kind: Some(Kind::Int64Value(*v as i64)),
            },
            PValue::I64(v) => pb::ScalarValue {
                kind: Some(Kind::Int64Value(*v)),
            },
            PValue::U8(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(*v as u64)),
            },
            PValue::U16(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(*v as u64)),
            },
            PValue::U32(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(*v as u64)),
            },
            PValue::U64(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(*v)),
            },
            PValue::F16(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(v.to_bits() as u64)),
            },
            PValue::F32(v) => pb::ScalarValue {
                kind: Some(Kind::F32Value(*v)),
            },
            PValue::F64(v) => pb::ScalarValue {
                kind: Some(Kind::F64Value(*v)),
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

        let value = ScalarValue::try_from(
            value
                .value
                .as_ref()
                .ok_or_else(|| vortex_err!(InvalidSerde: "Scalar missing value"))?,
        )?;

        Ok(Self { dtype, value })
    }
}

impl TryFrom<&pb::ScalarValue> for ScalarValue {
    type Error = VortexError;

    fn try_from(value: &pb::ScalarValue) -> Result<Self, Self::Error> {
        let kind = value
            .kind
            .as_ref()
            .ok_or_else(|| vortex_err!(InvalidSerde: "ScalarValue missing kind"))?;

        match kind {
            Kind::NullValue(_) => Ok(ScalarValue(InnerScalarValue::Null)),
            Kind::BoolValue(v) => Ok(ScalarValue(InnerScalarValue::Bool(*v))),
            Kind::Int64Value(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::I64(*v)))),
            Kind::Uint64Value(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::U64(*v)))),
            Kind::F32Value(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::F32(*v)))),
            Kind::F64Value(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::F64(*v)))),
            Kind::StringValue(v) => Ok(ScalarValue(InnerScalarValue::BufferString(Arc::new(
                BufferString::from(v.clone()),
            )))),
            Kind::BytesValue(v) => Ok(ScalarValue(InnerScalarValue::Buffer(Arc::new(
                ByteBuffer::from(v.clone()),
            )))),
            Kind::ListValue(v) => {
                let mut values = Vec::with_capacity(v.values.len());
                for elem in v.values.iter() {
                    values.push(elem.try_into()?);
                }
                Ok(ScalarValue(InnerScalarValue::List(values.into())))
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use vortex_buffer::BufferString;
    use vortex_dtype::PType::{self, I32};
    use vortex_dtype::half::f16;
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
            ScalarValue(InnerScalarValue::Buffer(Arc::new(vec![1, 2, 3].into()))),
        ));
    }

    #[test]
    fn test_buffer_string() {
        round_trip(Scalar::new(
            DType::Utf8(Nullability::Nullable),
            ScalarValue(InnerScalarValue::BufferString(Arc::new(
                BufferString::from("hello".to_string()),
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
                    ScalarValue(InnerScalarValue::Primitive(42i32.into())),
                    ScalarValue(InnerScalarValue::Primitive(43i32.into())),
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

    #[test]
    fn test_i8() {
        round_trip(Scalar::new(
            DType::Primitive(PType::I8, Nullability::Nullable),
            ScalarValue(InnerScalarValue::Primitive(i8::MIN.into())),
        ));

        round_trip(Scalar::new(
            DType::Primitive(PType::I8, Nullability::Nullable),
            ScalarValue(InnerScalarValue::Primitive(0i8.into())),
        ));

        round_trip(Scalar::new(
            DType::Primitive(PType::I8, Nullability::Nullable),
            ScalarValue(InnerScalarValue::Primitive(i8::MAX.into())),
        ));
    }
}

#[cfg(test)]
mod tests {
    use half::f16;
    use rstest::rstest;
    use vortex_dtype::{DType, DecimalDType, FieldDType, Nullability, PType, StructDType, half};

    use super::*;
    use crate::{Scalar, i256};

    #[rstest]
    #[case(Scalar::binary(ByteBuffer::copy_from(b"hello"), Nullability::NonNullable))]
    #[case(Scalar::utf8("hello", Nullability::NonNullable))]
    #[case(Scalar::primitive(1u8, Nullability::NonNullable))]
    #[case(Scalar::primitive(
        f32::from_bits(u32::from_le_bytes([0xFFu8, 0x8A, 0xF9, 0xFF])),
        Nullability::NonNullable
    ))]
    #[case(Scalar::list(Arc::new(PType::U8.into()), vec![Scalar::primitive(1u8, Nullability::NonNullable)], Nullability::NonNullable
    ))]
    #[case(Scalar::struct_(DType::Struct(
        Arc::new(StructDType::from_iter([
            ("a", FieldDType::from(DType::Primitive(PType::U32, Nullability::NonNullable))),
            ("b", FieldDType::from(DType::Primitive(PType::F16, Nullability::NonNullable))),
        ])),
        Nullability::NonNullable),
        vec![
            Scalar::primitive(23592960, Nullability::NonNullable),
            Scalar::primitive(f16::from_f32(2.6584664e36f32), Nullability::NonNullable),
        ],
    ))]
    #[case(Scalar::struct_(DType::Struct(
        Arc::new(StructDType::from_iter([
            ("a", FieldDType::from(DType::Primitive(PType::U64, Nullability::NonNullable))),
            ("b", FieldDType::from(DType::Primitive(PType::F32, Nullability::NonNullable))),
            ("c", FieldDType::from(DType::Primitive(PType::F16, Nullability::NonNullable))),
        ])),
        Nullability::NonNullable),
        vec![
            Scalar::primitive(415118687234i64, Nullability::NonNullable),
            Scalar::primitive(2.6584664e36f32, Nullability::NonNullable),
            Scalar::primitive(f16::from_f32(2.6584664e36f32), Nullability::NonNullable),
        ],
    ))]
    #[case(Scalar::decimal(
        DecimalValue::I256(i256::from_i128(12345643673471)),
        DecimalDType::new(10, 2),
        Nullability::NonNullable
    ))]
    #[case(Scalar::decimal(
        DecimalValue::I16(23412),
        DecimalDType::new(3, 2),
        Nullability::NonNullable
    ))]
    fn test_scalar_value_serde_roundtrip(#[case] scalar: Scalar) {
        let written = scalar.value.to_protobytes::<Vec<u8>>();
        let scalar_read_back = ScalarValue::from_protobytes(&written).unwrap();
        assert_eq!(
            scalar,
            Scalar::new(scalar.dtype().clone(), scalar_read_back)
        );
    }
}

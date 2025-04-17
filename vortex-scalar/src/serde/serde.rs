use std::fmt::Formatter;
use std::sync::Arc;

use serde::de::{Error, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use vortex_buffer::{BufferString, ByteBuffer};

use crate::pvalue::PValue;
use crate::{InnerScalarValue, ScalarValue};

impl Serialize for ScalarValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl Serialize for InnerScalarValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Null => ().serialize(serializer),
            Self::Bool(b) => b.serialize(serializer),
            Self::Primitive(p) => p.serialize(serializer),
            // NOTE: we explicitly handle the serialization of bytes, strings and lists so as not
            //  to create ambiguities amongst them. The serde data model has specific representations
            //  of binary data, UTF-8 strings and sequences.
            //  See https://serde.rs/data-model.html.
            Self::Buffer(buffer) => serializer.serialize_bytes(buffer.as_slice()),
            Self::BufferString(buffer) => serializer.serialize_str(buffer.as_str()),
            Self::List(l) => {
                let mut seq = serializer.serialize_seq(Some(l.len()))?;
                for item in l.iter() {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ScalarValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ScalarValueVisitor;
        impl<'v> Visitor<'v> for ScalarValueVisitor {
            type Value = ScalarValue;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                write!(formatter, "a scalar data value")
            }

            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Bool(v)))
            }

            fn visit_i8<E>(self, v: i8) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Primitive(PValue::I8(v))))
            }

            fn visit_i16<E>(self, v: i16) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Primitive(PValue::I16(v))))
            }

            fn visit_i32<E>(self, v: i32) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Primitive(PValue::I32(v))))
            }

            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Primitive(PValue::I64(v))))
            }

            fn visit_u8<E>(self, v: u8) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Primitive(PValue::U8(v))))
            }

            fn visit_u16<E>(self, v: u16) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Primitive(PValue::U16(v))))
            }

            fn visit_u32<E>(self, v: u32) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Primitive(PValue::U32(v))))
            }

            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Primitive(PValue::U64(v))))
            }

            fn visit_f32<E>(self, v: f32) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Primitive(PValue::F32(v))))
            }

            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Primitive(PValue::F64(v))))
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::BufferString(Arc::new(
                    BufferString::from(v.to_string()),
                ))))
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Buffer(Arc::new(
                    ByteBuffer::from(v.to_vec()),
                ))))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Null))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'v>,
            {
                let mut elems = Vec::with_capacity(seq.size_hint().unwrap_or_default());
                while let Some(e) = seq.next_element::<ScalarValue>()? {
                    elems.push(e);
                }
                Ok(ScalarValue(InnerScalarValue::List(elems.into())))
            }
        }

        deserializer.deserialize_any(ScalarValueVisitor)
    }
}

impl Serialize for PValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::U8(v) => serializer.serialize_u8(*v),
            Self::U16(v) => serializer.serialize_u16(*v),
            Self::U32(v) => serializer.serialize_u32(*v),
            Self::U64(v) => serializer.serialize_u64(*v),
            Self::I8(v) => serializer.serialize_i8(*v),
            Self::I16(v) => serializer.serialize_i16(*v),
            Self::I32(v) => serializer.serialize_i32(*v),
            Self::I64(v) => serializer.serialize_i64(*v),
            // NOTE(ngates): f16's are serialized bit-wise as u16.
            Self::F16(v) => serializer.serialize_u16(v.to_bits()),
            Self::F32(v) => serializer.serialize_u32(v.to_bits()),
            Self::F64(v) => serializer.serialize_u64(v.to_bits()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use flexbuffers::{FlexbufferSerializer, Reader};
    use rstest::rstest;
    use vortex_dtype::half::f16;
    use vortex_dtype::{DType, FieldDType, Nullability, PType, StructDType};

    use super::*;
    use crate::Scalar;

    #[rstest]
    #[case(Scalar::binary(ByteBuffer::copy_from(b"hello"), Nullability::NonNullable))]
    #[case(Scalar::utf8("hello", Nullability::NonNullable))]
    #[case(Scalar::primitive(1u8, Nullability::NonNullable))]
    #[case(Scalar::primitive(f32::from_bits(u32::from_le_bytes([0xFFu8, 0x8A, 0xF9, 0xFF])), Nullability::NonNullable))]
    #[case(Scalar::list(Arc::new(PType::U8.into()), vec![Scalar::primitive(1u8, Nullability::NonNullable)], Nullability::NonNullable))]
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
    fn test_scalar_value_serde_roundtrip(#[case] scalar: Scalar) {
        let mut serializer = FlexbufferSerializer::new();
        scalar.value.serialize(&mut serializer).unwrap();
        let written = serializer.take_buffer();
        let reader = Reader::get_root(written.as_ref()).unwrap();
        let scalar_read_back = ScalarValue::deserialize(reader).unwrap();
        assert_eq!(
            scalar,
            Scalar::new(scalar.dtype().clone(), scalar_read_back)
        );
    }
}

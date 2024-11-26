use std::fmt::Formatter;

use serde::de::{Error, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use vortex_buffer::BufferString;

use crate::pvalue::PValue;
use crate::value::{InnerScalarValue, ScalarValue};

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
            Self::Buffer(buffer) => buffer.as_ref().serialize(serializer),
            Self::BufferString(buffer) => buffer.as_str().serialize(serializer),
            Self::List(l) => l.serialize(serializer),
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
                Ok(ScalarValue(InnerScalarValue::BufferString(
                    BufferString::from(v.to_string()),
                )))
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ScalarValue(InnerScalarValue::Buffer(v.to_vec().into())))
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
                let mut elems = vec![];
                while let Some(e) = seq.next_element::<ScalarValue>()? {
                    elems.push(e);
                }
                Ok(ScalarValue(InnerScalarValue::List(
                    elems.iter().map(|x| x.0.clone()).collect(),
                )))
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
            Self::F32(v) => serializer.serialize_f32(*v),
            Self::F64(v) => serializer.serialize_f64(*v),
        }
    }
}

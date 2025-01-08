#![allow(clippy::disallowed_types)]
use std::collections::HashMap;
use std::io::Cursor;

use apache_avro::types::Value;
use apache_avro::{from_avro_datum, to_avro_datum, BigDecimal, Decimal, Duration, Schema};
use uuid::Uuid;
pub use vortex_avro_derive::{FromAvro, ToAvro};
use vortex_error::{vortex_err, VortexError, VortexResult};

mod array;
mod option;
mod prim;
mod string;

/// AvroValue is based on `Value` from the Avro crate, but without the blanket impls. This is so we have control over how the
/// conversions for primitives are implemented.
#[derive(Debug)]
pub enum AvroValue {
    Null,
    Boolean(bool),
    Int(i32),
    Float(f32),
    Double(f64),
    Bytes(Vec<u8>),
    Fixed(usize, Vec<u8>),
    Enum(u32, String),
    Union(u32, Box<AvroValue>),
    Array(Vec<AvroValue>),
    Map(HashMap<String, AvroValue>),
    Record(Vec<(String, AvroValue)>),
    Date(i32),
    Decimal(Decimal),
    BigDecimal(BigDecimal),
    TimeMillis(i32),
    TimeMicros(i64),
    TimestampMillis(i64),
    TimestampMicros(i64),
    TimestampNanos(i64),
    LocalTimestampMillis(i64),
    LocalTimestampMicros(i64),
    LocalTimestampNanos(i64),
    Duration(Duration),
    Uuid(Uuid),
    Long(i64),
    String(String),
}

// Helper conversion into our AvroValue type from upstream `apache_avro::Value`.
impl From<Value> for AvroValue {
    fn from(value: Value) -> Self {
        match value {
            Value::Long(i) => AvroValue::Long(i),
            Value::String(s) => AvroValue::String(s),
            Value::Int(i) => AvroValue::Int(i),
            Value::Null => AvroValue::Null,
            Value::Boolean(b) => AvroValue::Boolean(b),
            Value::Float(f) => AvroValue::Float(f),
            Value::Double(d) => AvroValue::Double(d),
            Value::Bytes(b) => AvroValue::Bytes(b),
            Value::Fixed(size, bytes) => AvroValue::Fixed(size, bytes),
            Value::Enum(i, s) => AvroValue::Enum(i, s),
            Value::Union(i, v) => AvroValue::Union(i, Box::new((*v).into())),
            Value::Array(items) => {
                AvroValue::Array(items.into_iter().map(AvroValue::from).collect())
            }
            Value::Map(items) => {
                AvroValue::Map(items.into_iter().map(|(k, v)| (k, v.into())).collect())
            }
            Value::Record(items) => {
                AvroValue::Record(items.into_iter().map(|(k, v)| (k, v.into())).collect())
            }
            Value::Date(d) => AvroValue::Date(d),
            Value::Decimal(d) => AvroValue::Decimal(d),
            Value::BigDecimal(d) => AvroValue::BigDecimal(d),
            Value::TimeMillis(t) => AvroValue::TimeMillis(t),
            Value::TimeMicros(t) => AvroValue::TimeMicros(t),
            Value::TimestampMillis(t) => AvroValue::TimestampMillis(t),
            Value::TimestampMicros(t) => AvroValue::TimestampMicros(t),
            Value::TimestampNanos(t) => AvroValue::TimestampNanos(t),
            Value::LocalTimestampMillis(t) => AvroValue::LocalTimestampMillis(t),
            Value::LocalTimestampMicros(t) => AvroValue::LocalTimestampMicros(t),
            Value::LocalTimestampNanos(t) => AvroValue::LocalTimestampNanos(t),
            Value::Duration(d) => AvroValue::Duration(d),
            Value::Uuid(u) => AvroValue::Uuid(u),
        }
    }
}

// Helper conversion into upstream `apache_avro::Value` from our `AvroValue` type.
impl From<AvroValue> for Value {
    fn from(value: AvroValue) -> Self {
        match value {
            AvroValue::Long(i) => Value::Long(i),
            AvroValue::String(s) => Value::String(s),
            AvroValue::Int(i) => Value::Int(i),
            AvroValue::Null => Value::Null,
            AvroValue::Boolean(b) => Value::Boolean(b),
            AvroValue::Float(f) => Value::Float(f),
            AvroValue::Double(d) => Value::Double(d),
            AvroValue::Bytes(b) => Value::Bytes(b),
            AvroValue::Fixed(size, bytes) => Value::Fixed(size, bytes),
            AvroValue::Enum(i, s) => Value::Enum(i, s),
            AvroValue::Union(i, v) => Value::Union(i, Box::new((*v).into())),
            AvroValue::Array(items) => Value::Array(items.into_iter().map(Value::from).collect()),
            AvroValue::Map(items) => {
                Value::Map(items.into_iter().map(|(k, v)| (k, v.into())).collect())
            }
            AvroValue::Record(items) => {
                Value::Record(items.into_iter().map(|(k, v)| (k, v.into())).collect())
            }
            AvroValue::Date(d) => Value::Date(d),
            AvroValue::Decimal(d) => Value::Decimal(d),
            AvroValue::BigDecimal(d) => Value::BigDecimal(d),
            AvroValue::TimeMillis(t) => Value::TimeMillis(t),
            AvroValue::TimeMicros(t) => Value::TimeMicros(t),
            AvroValue::TimestampMillis(t) => Value::TimestampMillis(t),
            AvroValue::TimestampMicros(t) => Value::TimestampMicros(t),
            AvroValue::TimestampNanos(t) => Value::TimestampNanos(t),
            AvroValue::LocalTimestampMillis(t) => Value::LocalTimestampMillis(t),
            AvroValue::LocalTimestampMicros(t) => Value::LocalTimestampMicros(t),
            AvroValue::LocalTimestampNanos(t) => Value::LocalTimestampNanos(t),
            AvroValue::Duration(d) => Value::Duration(d),
            AvroValue::Uuid(u) => Value::Uuid(u),
        }
    }
}

/// Types that can be converted into an Avro value type.
///
/// Types must provide a conversion into [`AvroValue`], the type-erased value type
/// for the Avro binary format.
///
/// Additionally, types must provide a schema that can be used to write the type to the Avro binary format.
pub trait ToAvro: Into<AvroValue> {
    // TODO(aduffy): just have one schema instead of read/write.
    fn write_schema() -> Schema;
}

/// Types that can be deserialized from an Avro binary format.
///
/// Types must provide a conversion from [`AvroValue`], the type-erased value type
/// for the Avro binary format.
///
/// Additionally, types must provide a schema that can be used to read the type from the Avro binary format.
pub trait FromAvro: TryFrom<AvroValue, Error = VortexError> {
    /// Retrieve the Avro schema that is used to read this type from the Avro binary format.
    fn read_schema() -> Schema;
}

/// Convert a type into the Avro binary format.
///
/// This function will return an error if the type cannot be converted into the Avro binary format.
pub fn to_avro_binary<T: ToAvro>(value: T) -> VortexResult<Vec<u8>> {
    let avro_value: AvroValue = value.into();
    to_avro_datum(&T::write_schema(), avro_value)
        .map_err(|err| vortex_err!("Failed to convert type to Avro binary format: {err}"))
}

/// Read into a type from the Avro binary format.
///
/// This function will return an error if the type cannot be read from the Avro binary format.
pub fn from_avro_binary<T: FromAvro>(schema: &Schema, avro_bytes: Vec<u8>) -> VortexResult<T> {
    let value = from_avro_datum(schema, &mut Cursor::new(avro_bytes), None)
        .map_err(|err| vortex_err!("Failed to read type from Avro binary format: {err}"))?;
    <T as TryFrom<AvroValue>>::try_from(value.into())
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! test_roundtrip {
        ($name:ident => $ty:ty, $value:expr) => {
            #[test]
            fn $name() {
                let value: $ty = $value;
                let avro_bytes = to_avro_binary(value).expect("to_avro_binary");
                let value_read: $ty = from_avro_binary::<$ty>(&<$ty>::read_schema(), avro_bytes)
                    .expect("from_avro_binary");
                assert_eq!($value, value_read);
            }
        };
    }

    test_roundtrip!(test_u8 => u8, u8::MAX);
    test_roundtrip!(test_u16 => u16, u16::MAX);
    test_roundtrip!(test_u32 => u32, u32::MAX);
    test_roundtrip!(test_u64 => u64, u64::MAX);
    test_roundtrip!(test_i8 => i8, i8::MAX);
    test_roundtrip!(test_i16 => i16, i16::MAX);
    test_roundtrip!(test_i32 => i32, i32::MAX);
    test_roundtrip!(test_i64 => i64, i64::MAX);
    test_roundtrip!(test_string_opt => Option<String>, Some("hello".to_string()));
    test_roundtrip!(test_string_opt_none => Option<String>, None::<String>);
    test_roundtrip!(test_u64_opt => Option<u64>, Some(u64::MAX));
    test_roundtrip!(test_u64_opt_none => Option<u64>, None::<u64>);
}

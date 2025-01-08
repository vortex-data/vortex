#![allow(clippy::disallowed_types)]
use std::collections::HashMap;
use std::io::Cursor;

use uuid::Uuid;
pub use vortex_avro_derive::{FromAvro, ToAvro};
use vortex_error::{vortex_err, VortexError, VortexResult};

mod array;
mod option;
mod prim;
mod string;
mod vec;

pub mod avro_private {
    pub use apache_avro::schema::*;
    pub use apache_avro::types::*;
    pub use apache_avro::*;
}

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
    Decimal(avro_private::Decimal),
    BigDecimal(avro_private::BigDecimal),
    TimeMillis(i32),
    TimeMicros(i64),
    TimestampMillis(i64),
    TimestampMicros(i64),
    TimestampNanos(i64),
    LocalTimestampMillis(i64),
    LocalTimestampMicros(i64),
    LocalTimestampNanos(i64),
    Duration(avro_private::Duration),
    Uuid(Uuid),
    Long(i64),
    String(String),
}

// Helper conversion into our AvroValue type from upstream `apache_avro::Value`.
impl From<avro_private::Value> for AvroValue {
    fn from(value: avro_private::Value) -> Self {
        match value {
            avro_private::Value::Long(i) => AvroValue::Long(i),
            avro_private::Value::String(s) => AvroValue::String(s),
            avro_private::Value::Int(i) => AvroValue::Int(i),
            avro_private::Value::Null => AvroValue::Null,
            avro_private::Value::Boolean(b) => AvroValue::Boolean(b),
            avro_private::Value::Float(f) => AvroValue::Float(f),
            avro_private::Value::Double(d) => AvroValue::Double(d),
            avro_private::Value::Bytes(b) => AvroValue::Bytes(b),
            avro_private::Value::Fixed(size, bytes) => AvroValue::Fixed(size, bytes),
            avro_private::Value::Enum(i, s) => AvroValue::Enum(i, s),
            avro_private::Value::Union(i, v) => AvroValue::Union(i, Box::new((*v).into())),
            avro_private::Value::Array(items) => {
                AvroValue::Array(items.into_iter().map(AvroValue::from).collect())
            }
            avro_private::Value::Map(items) => {
                AvroValue::Map(items.into_iter().map(|(k, v)| (k, v.into())).collect())
            }
            avro_private::Value::Record(items) => {
                AvroValue::Record(items.into_iter().map(|(k, v)| (k, v.into())).collect())
            }
            avro_private::Value::Date(d) => AvroValue::Date(d),
            avro_private::Value::Decimal(d) => AvroValue::Decimal(d),
            avro_private::Value::BigDecimal(d) => AvroValue::BigDecimal(d),
            avro_private::Value::TimeMillis(t) => AvroValue::TimeMillis(t),
            avro_private::Value::TimeMicros(t) => AvroValue::TimeMicros(t),
            avro_private::Value::TimestampMillis(t) => AvroValue::TimestampMillis(t),
            avro_private::Value::TimestampMicros(t) => AvroValue::TimestampMicros(t),
            avro_private::Value::TimestampNanos(t) => AvroValue::TimestampNanos(t),
            avro_private::Value::LocalTimestampMillis(t) => AvroValue::LocalTimestampMillis(t),
            avro_private::Value::LocalTimestampMicros(t) => AvroValue::LocalTimestampMicros(t),
            avro_private::Value::LocalTimestampNanos(t) => AvroValue::LocalTimestampNanos(t),
            avro_private::Value::Duration(d) => AvroValue::Duration(d),
            avro_private::Value::Uuid(u) => AvroValue::Uuid(u),
        }
    }
}

// Helper conversion into upstream `apache_avro::Value` from our `AvroValue` type.
impl From<AvroValue> for avro_private::Value {
    fn from(value: AvroValue) -> Self {
        match value {
            AvroValue::Long(i) => avro_private::Value::Long(i),
            AvroValue::String(s) => avro_private::Value::String(s),
            AvroValue::Int(i) => avro_private::Value::Int(i),
            AvroValue::Null => avro_private::Value::Null,
            AvroValue::Boolean(b) => avro_private::Value::Boolean(b),
            AvroValue::Float(f) => avro_private::Value::Float(f),
            AvroValue::Double(d) => avro_private::Value::Double(d),
            AvroValue::Bytes(b) => avro_private::Value::Bytes(b),
            AvroValue::Fixed(size, bytes) => avro_private::Value::Fixed(size, bytes),
            AvroValue::Enum(i, s) => avro_private::Value::Enum(i, s),
            AvroValue::Union(i, v) => avro_private::Value::Union(i, Box::new((*v).into())),
            AvroValue::Array(items) => avro_private::Value::Array(
                items.into_iter().map(avro_private::Value::from).collect(),
            ),
            AvroValue::Map(items) => {
                avro_private::Value::Map(items.into_iter().map(|(k, v)| (k, v.into())).collect())
            }
            AvroValue::Record(items) => {
                avro_private::Value::Record(items.into_iter().map(|(k, v)| (k, v.into())).collect())
            }
            AvroValue::Date(d) => avro_private::Value::Date(d),
            AvroValue::Decimal(d) => avro_private::Value::Decimal(d),
            AvroValue::BigDecimal(d) => avro_private::Value::BigDecimal(d),
            AvroValue::TimeMillis(t) => avro_private::Value::TimeMillis(t),
            AvroValue::TimeMicros(t) => avro_private::Value::TimeMicros(t),
            AvroValue::TimestampMillis(t) => avro_private::Value::TimestampMillis(t),
            AvroValue::TimestampMicros(t) => avro_private::Value::TimestampMicros(t),
            AvroValue::TimestampNanos(t) => avro_private::Value::TimestampNanos(t),
            AvroValue::LocalTimestampMillis(t) => avro_private::Value::LocalTimestampMillis(t),
            AvroValue::LocalTimestampMicros(t) => avro_private::Value::LocalTimestampMicros(t),
            AvroValue::LocalTimestampNanos(t) => avro_private::Value::LocalTimestampNanos(t),
            AvroValue::Duration(d) => avro_private::Value::Duration(d),
            AvroValue::Uuid(u) => avro_private::Value::Uuid(u),
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
    fn write_schema() -> avro_private::Schema;
}

/// Types that can be deserialized from an Avro binary format.
///
/// Types must provide a conversion from [`AvroValue`], the type-erased value type
/// for the Avro binary format.
///
/// Additionally, types must provide a schema that can be used to read the type from the Avro binary format.
pub trait FromAvro: TryFrom<AvroValue, Error = VortexError> {
    /// Retrieve the Avro schema that is used to read this type from the Avro binary format.
    fn read_schema() -> avro_private::Schema;
}

/// Convert a type into the Avro binary format.
///
/// This function will return an error if the type cannot be converted into the Avro binary format.
pub fn to_avro_binary<T: ToAvro>(value: T) -> VortexResult<Vec<u8>> {
    let avro_value: AvroValue = value.into();
    avro_private::to_avro_datum(&T::write_schema(), avro_value)
        .map_err(|err| vortex_err!("Failed to convert type to Avro binary format: {err}"))
}

/// Read into a type from the Avro binary format.
///
/// This function will return an error if the type cannot be read from the Avro binary format.
pub fn from_avro_binary<T: FromAvro>(
    schema: &avro_private::Schema,
    avro_bytes: Vec<u8>,
) -> VortexResult<T> {
    let value = avro_private::from_avro_datum(schema, &mut Cursor::new(avro_bytes), None)
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

use vortex_error::{vortex_err, VortexError};

use crate::{AvroValue, FromAvro, ToAvro};

impl From<String> for AvroValue {
    fn from(value: String) -> Self {
        AvroValue::String(value)
    }
}

impl ToAvro for String {
    fn write_schema() -> crate::avro_private::Schema {
        crate::avro_private::Schema::String
    }
}

impl TryFrom<AvroValue> for String {
    type Error = VortexError;

    fn try_from(value: AvroValue) -> Result<Self, Self::Error> {
        if let AvroValue::String(s) = value {
            Ok(s)
        } else {
            Err(vortex_err!("Expected value to be a String but it was not"))
        }
    }
}

impl FromAvro for String {
    fn read_schema() -> crate::avro_private::Schema {
        crate::avro_private::Schema::String
    }
}

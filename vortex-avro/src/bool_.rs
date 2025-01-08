use vortex_error::{vortex_bail, VortexError};

use crate::{AvroValue, FromAvro, ToAvro};

impl From<bool> for AvroValue {
    fn from(value: bool) -> Self {
        AvroValue::Boolean(value)
    }
}

impl TryFrom<AvroValue> for bool {
    type Error = VortexError;

    fn try_from(value: AvroValue) -> Result<Self, Self::Error> {
        if let AvroValue::Boolean(v) = value {
            Ok(v)
        } else {
            vortex_bail!("Expected value to be a Boolean but it was not")
        }
    }
}

impl FromAvro for bool {
    fn read_schema() -> crate::avro_private::Schema {
        crate::avro_private::Schema::Boolean
    }
}

impl ToAvro for bool {
    fn write_schema() -> crate::avro_private::Schema {
        crate::avro_private::Schema::Boolean
    }
}

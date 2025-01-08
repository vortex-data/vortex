//
// Fixed-size primitive arrays.
//

use apache_avro::schema::ArraySchema;
use apache_avro::Schema;
use vortex_error::{vortex_bail, VortexError};

use crate::{AvroValue, FromAvro, ToAvro};

// ToAvro
impl<T: Into<AvroValue>> From<Vec<T>> for AvroValue {
    fn from(value: Vec<T>) -> Self {
        AvroValue::Array(value.into_iter().map(Into::into).collect())
    }
}

impl<T: ToAvro> ToAvro for Vec<T> {
    fn write_schema() -> Schema {
        Schema::Array(ArraySchema {
            items: Box::new(T::write_schema()),
            attributes: Default::default(),
        })
    }
}

// FromAvro
impl<T: FromAvro> TryFrom<AvroValue> for Vec<T> {
    type Error = VortexError;

    fn try_from(value: AvroValue) -> Result<Self, Self::Error> {
        let AvroValue::Array(items) = value else {
            vortex_bail!("Expected value to be an array but it was not");
        };
        let items: Vec<T> = items
            .into_iter()
            .map(T::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(items)
    }
}

impl<T: FromAvro> FromAvro for Vec<T> {
    fn read_schema() -> Schema {
        Schema::Array(ArraySchema {
            items: Box::new(T::read_schema()),
            attributes: Default::default(),
        })
    }
}

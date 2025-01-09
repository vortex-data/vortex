use vortex_error::{vortex_bail, vortex_err, VortexError};

use crate::{AvroValue, FromAvro, ToAvro};

// ToAvro
impl<const N: usize, T: Into<AvroValue>> From<[T; N]> for AvroValue {
    fn from(value: [T; N]) -> Self {
        AvroValue::Array(value.into_iter().map(Into::into).collect())
    }
}

impl<const N: usize, T: ToAvro> ToAvro for [T; N] {
    fn write_schema(prefix: impl AsRef<str>) -> crate::avro_private::Schema {
        crate::avro_private::Schema::Array(crate::avro_private::ArraySchema {
            items: Box::new(T::write_schema(prefix)),
            attributes: Default::default(),
        })
    }
}

// FromAvro
impl<const N: usize, T: FromAvro> TryFrom<AvroValue> for [T; N] {
    type Error = VortexError;

    fn try_from(value: AvroValue) -> Result<Self, Self::Error> {
        let AvroValue::Array(items) = value else {
            vortex_bail!("Expected value to be an array but it was not");
        };
        let items: Vec<T> = items
            .into_iter()
            .map(T::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        <[T; N]>::try_from(items).map_err(|items| {
            vortex_err!(
                "Expected fixed-size array of length {N}, was {}",
                items.len()
            )
        })
    }
}

impl<const N: usize, T: FromAvro> FromAvro for [T; N] {
    fn read_schema() -> crate::avro_private::Schema {
        crate::avro_private::Schema::Array(crate::avro_private::ArraySchema {
            items: Box::new(T::read_schema()),
            attributes: Default::default(),
        })
    }
}

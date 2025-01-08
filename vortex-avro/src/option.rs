use apache_avro::schema::UnionSchema;
use apache_avro::Schema;
use vortex_error::vortex_bail;

use crate::{AvroValue, FromAvro, ToAvro};

impl<T> From<Option<T>> for AvroValue
where
    T: Into<AvroValue>,
{
    fn from(value: Option<T>) -> Self {
        match value {
            Some(v) => AvroValue::Union(1, Box::new(v.into())),
            None => AvroValue::Union(0, Box::new(AvroValue::Null)),
        }
    }
}

impl<T> TryFrom<AvroValue> for Option<T>
where
    T: FromAvro,
{
    type Error = T::Error;

    fn try_from(value: AvroValue) -> Result<Self, Self::Error> {
        if let AvroValue::Union(idx, value) = value {
            if idx == 0 {
                Ok(None)
            } else {
                Ok(Some(T::try_from(*value)?))
            }
        } else {
            vortex_bail!("Option<T> Avro binary value must be a union with a null and the type T")
        }
    }
}

impl<T> FromAvro for Option<T>
where
    T: FromAvro,
{
    #[allow(clippy::expect_used)]
    fn read_schema() -> Schema {
        Schema::Union(
            UnionSchema::new(vec![Schema::Null, T::read_schema()]).expect("Option<T> schema"),
        )
    }
}

impl<T> ToAvro for Option<T>
where
    T: ToAvro,
{
    #[allow(clippy::expect_used)]
    fn write_schema() -> Schema {
        Schema::Union(
            UnionSchema::new(vec![Schema::Null, T::write_schema()]).expect("Option<T> schema"),
        )
    }
}

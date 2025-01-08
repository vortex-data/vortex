use vortex_error::{vortex_err, VortexError};

use crate::{AvroValue, FromAvro};

macro_rules! impl_primitive {
    ($ty:ty, $inner:ty, $value_variant:path, $schema_variant:path) => {
        impl From<$ty> for AvroValue {
            fn from(value: $ty) -> Self {
                $value_variant(value as $inner)
            }
        }

        impl $crate::ToAvro for $ty {
            fn write_schema() -> $crate::avro_private::Schema {
                $schema_variant
            }
        }

        impl TryFrom<AvroValue> for $ty {
            type Error = VortexError;

            fn try_from(value: AvroValue) -> Result<Self, Self::Error> {
                if let $value_variant(v) = value.into() {
                    Ok(<$inner>::try_from(v)? as $ty)
                } else {
                    Err(vortex_err!(
                        "Expected value to be a {} but it was not",
                        stringify!($value_variant)
                    ))
                }
            }
        }

        impl FromAvro for $ty {
            fn read_schema() -> $crate::avro_private::Schema {
                $schema_variant
            }
        }
    };
}

impl_primitive!(i8, i32, AvroValue::Int, crate::avro_private::Schema::Int);
impl_primitive!(i16, i32, AvroValue::Int, crate::avro_private::Schema::Int);
impl_primitive!(i32, i32, AvroValue::Int, crate::avro_private::Schema::Int);
impl_primitive!(u8, i32, AvroValue::Int, crate::avro_private::Schema::Int);
impl_primitive!(u16, i32, AvroValue::Int, crate::avro_private::Schema::Int);
impl_primitive!(u32, i32, AvroValue::Int, crate::avro_private::Schema::Int);
impl_primitive!(i64, i64, AvroValue::Long, crate::avro_private::Schema::Long);
impl_primitive!(u64, i64, AvroValue::Long, crate::avro_private::Schema::Long);
// TODO(aduffy): f16 support?
impl_primitive!(
    f32,
    f32,
    AvroValue::Float,
    crate::avro_private::Schema::Float
);
impl_primitive!(
    f64,
    f64,
    AvroValue::Double,
    crate::avro_private::Schema::Double
);

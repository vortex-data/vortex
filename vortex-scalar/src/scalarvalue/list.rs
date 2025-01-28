use std::sync::Arc;

use vortex_buffer::{BufferString, ByteBuffer};
use vortex_dtype::half::f16;
use vortex_error::{vortex_err, VortexError};

use crate::scalarvalue::InnerScalarValue;
use crate::ScalarValue;

impl<'a, T: for<'b> TryFrom<&'b ScalarValue, Error = VortexError>> TryFrom<&'a ScalarValue>
    for Vec<T>
{
    type Error = VortexError;

    fn try_from(value: &'a ScalarValue) -> Result<Self, Self::Error> {
        let value = value
            .as_list()?
            .ok_or_else(|| vortex_err!("Can't convert non list scalar to vec"))?;

        value.iter().map(|v| T::try_from(v)).collect()
    }
}

macro_rules! from_vec_for_scalar_value {
    ($T:ty) => {
        impl From<Vec<$T>> for ScalarValue {
            fn from(value: Vec<$T>) -> Self {
                ScalarValue(InnerScalarValue::List(
                    value
                        .into_iter()
                        .map(ScalarValue::from)
                        .collect::<Arc<[_]>>(),
                ))
            }
        }
    };
}

// no From<Vec<u8>> because it could either be a List or a Buffer
from_vec_for_scalar_value!(u16);
from_vec_for_scalar_value!(u32);
from_vec_for_scalar_value!(u64);
from_vec_for_scalar_value!(usize); // For usize only, we implicitly cast for better ergonomics.
from_vec_for_scalar_value!(i8);
from_vec_for_scalar_value!(i16);
from_vec_for_scalar_value!(i32);
from_vec_for_scalar_value!(i64);
from_vec_for_scalar_value!(f16);
from_vec_for_scalar_value!(f32);
from_vec_for_scalar_value!(f64);
from_vec_for_scalar_value!(String);
from_vec_for_scalar_value!(BufferString);
from_vec_for_scalar_value!(ByteBuffer);

use paste::paste;
use vortex_dtype::half::f16;
use vortex_error::{VortexError, vortex_err};

use crate::ScalarValue;
use crate::scalarvalue::InnerScalarValue;

macro_rules! primitive_scalar {
    ($T:ty) => {
        impl TryFrom<&ScalarValue> for $T {
            type Error = VortexError;

            fn try_from(value: &ScalarValue) -> Result<Self, Self::Error> {
                <Option<$T>>::try_from(value)?
                    .ok_or_else(|| vortex_err!("Can't extract present value from null scalar"))
            }
        }

        impl TryFrom<&ScalarValue> for Option<$T> {
            type Error = VortexError;

            fn try_from(value: &ScalarValue) -> Result<Self, Self::Error> {
                paste! {
                    Ok(value.as_pvalue()?.and_then(|v| v.[<as_ $T>]()))
                }
            }
        }

        impl From<$T> for ScalarValue {
            fn from(value: $T) -> Self {
                ScalarValue(InnerScalarValue::Primitive(value.into()))
            }
        }
    };
}

primitive_scalar!(u8);
primitive_scalar!(u16);
primitive_scalar!(u32);
primitive_scalar!(u64);
primitive_scalar!(i8);
primitive_scalar!(i16);
primitive_scalar!(i32);
primitive_scalar!(i64);
primitive_scalar!(f16);
primitive_scalar!(f32);
primitive_scalar!(f64);

/// Read a scalar as usize. For usize only, we implicitly cast for better ergonomics.
impl TryFrom<&ScalarValue> for usize {
    type Error = VortexError;

    fn try_from(value: &ScalarValue) -> Result<Self, Self::Error> {
        let prim = value
            .as_pvalue()?
            .and_then(|v| v.as_u64())
            .ok_or_else(|| vortex_err!("cannot convert Null to usize"))?;
        Ok(usize::try_from(prim)?)
    }
}

/// Read a scalar as usize. For usize only, we implicitly cast for better ergonomics.
impl From<usize> for ScalarValue {
    fn from(value: usize) -> Self {
        ScalarValue(InnerScalarValue::Primitive((value as u64).into()))
    }
}

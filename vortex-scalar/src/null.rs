use vortex_error::VortexError;

use crate::{Scalar, ScalarValue};

impl TryFrom<&Scalar> for () {
    type Error = VortexError;

    fn try_from(scalar: &Scalar) -> Result<Self, Self::Error> {
        scalar.value().as_null()
    }
}

impl TryFrom<Scalar> for () {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        <()>::try_from(&scalar)
    }
}

impl TryFrom<&ScalarValue> for () {
    type Error = VortexError;

    fn try_from(value: &ScalarValue) -> Result<Self, Self::Error> {
        value.as_null()
    }
}

impl TryFrom<ScalarValue> for () {
    type Error = VortexError;

    fn try_from(value: ScalarValue) -> Result<Self, Self::Error> {
        <()>::try_from(&value)
    }
}

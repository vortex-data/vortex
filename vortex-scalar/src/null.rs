use vortex_error::VortexError;

use crate::Scalar;

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

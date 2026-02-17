// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversions from scalars into other types.

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::scalar::BinaryScalar;
use crate::scalar::BoolScalar;
use crate::scalar::DecimalScalar;
use crate::scalar::ExtScalar;
use crate::scalar::ListScalar;
use crate::scalar::PrimitiveScalar;
use crate::scalar::Scalar;
use crate::scalar::StructScalar;
use crate::scalar::Utf8Scalar;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Typed scalar conversions.
//
// These delegate to the `as_*_opt()` methods on [`Scalar`].
////////////////////////////////////////////////////////////////////////////////////////////////////

impl<'a> TryFrom<&'a Scalar> for BoolScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> VortexResult<Self> {
        value
            .as_bool_opt()
            .ok_or_else(|| vortex_err!("Expected bool scalar, found {}", value.dtype()))
    }
}

impl<'a> TryFrom<&'a Scalar> for PrimitiveScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> VortexResult<Self> {
        value
            .as_primitive_opt()
            .ok_or_else(|| vortex_err!("Expected primitive scalar, found {}", value.dtype()))
    }
}

impl<'a> TryFrom<&'a Scalar> for DecimalScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> VortexResult<Self> {
        value
            .as_decimal_opt()
            .ok_or_else(|| vortex_err!("Expected decimal scalar, found {}", value.dtype()))
    }
}

impl<'a> TryFrom<&'a Scalar> for Utf8Scalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> VortexResult<Self> {
        value
            .as_utf8_opt()
            .ok_or_else(|| vortex_err!("Expected utf8 scalar, found {}", value.dtype()))
    }
}

impl<'a> TryFrom<&'a Scalar> for BinaryScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> VortexResult<Self> {
        value
            .as_binary_opt()
            .ok_or_else(|| vortex_err!("Expected binary scalar, found {}", value.dtype()))
    }
}

impl<'a> TryFrom<&'a Scalar> for StructScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> VortexResult<Self> {
        value
            .as_struct_opt()
            .ok_or_else(|| vortex_err!("Expected struct scalar, found {}", value.dtype()))
    }
}

impl<'a> TryFrom<&'a Scalar> for ListScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> VortexResult<Self> {
        value
            .as_list_opt()
            .ok_or_else(|| vortex_err!("Expected list scalar, found {}", value.dtype()))
    }
}

impl<'a> TryFrom<&'a Scalar> for ExtScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> VortexResult<Self> {
        value
            .as_extension_opt()
            .ok_or_else(|| vortex_err!("Expected extension scalar, found {}", value.dtype()))
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Boolean conversions.
////////////////////////////////////////////////////////////////////////////////////////////////////

impl TryFrom<&Scalar> for bool {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> VortexResult<Self> {
        <Option<bool>>::try_from(value)?
            .ok_or_else(|| vortex_err!("Can't extract present value from null scalar"))
    }
}

impl TryFrom<&Scalar> for Option<bool> {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> VortexResult<Self> {
        Ok(value
            .as_bool_opt()
            .ok_or_else(|| vortex_err!("Expected bool scalar, found {}", value.dtype()))?
            .value())
    }
}

impl TryFrom<Scalar> for bool {
    type Error = VortexError;

    fn try_from(value: Scalar) -> VortexResult<Self> {
        Self::try_from(&value)
    }
}

impl TryFrom<Scalar> for Option<bool> {
    type Error = VortexError;

    fn try_from(value: Scalar) -> VortexResult<Self> {
        Self::try_from(&value)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Binary conversions.
////////////////////////////////////////////////////////////////////////////////////////////////////

impl<'a> TryFrom<&'a Scalar> for ByteBuffer {
    type Error = VortexError;

    fn try_from(scalar: &'a Scalar) -> VortexResult<Self> {
        let binary = scalar
            .as_binary_opt()
            .ok_or_else(|| vortex_err!("Cannot extract buffer from non-buffer scalar"))?;

        binary
            .value()
            .cloned()
            .ok_or_else(|| vortex_err!("Cannot extract present value from null scalar"))
    }
}

impl<'a> TryFrom<&'a Scalar> for Option<ByteBuffer> {
    type Error = VortexError;

    fn try_from(scalar: &'a Scalar) -> VortexResult<Self> {
        Ok(scalar
            .as_binary_opt()
            .ok_or_else(|| vortex_err!("Cannot extract buffer from non-buffer scalar"))?
            .value()
            .cloned())
    }
}

impl TryFrom<Scalar> for ByteBuffer {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> VortexResult<Self> {
        Self::try_from(&scalar)
    }
}

impl TryFrom<Scalar> for Option<ByteBuffer> {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> VortexResult<Self> {
        Self::try_from(&scalar)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// UTF-8 conversions.
////////////////////////////////////////////////////////////////////////////////////////////////////

impl<'a> TryFrom<&'a Scalar> for String {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        Ok(BufferString::try_from(value)?.to_string())
    }
}

impl TryFrom<Scalar> for String {
    type Error = VortexError;

    fn try_from(value: Scalar) -> Result<Self, Self::Error> {
        Ok(BufferString::try_from(value)?.to_string())
    }
}

impl<'a> TryFrom<&'a Scalar> for BufferString {
    type Error = VortexError;

    fn try_from(scalar: &'a Scalar) -> VortexResult<Self> {
        <Option<BufferString>>::try_from(scalar)?
            .ok_or_else(|| vortex_err!("Can't extract present value from null scalar"))
    }
}

impl TryFrom<Scalar> for BufferString {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        Self::try_from(&scalar)
    }
}

impl<'a> TryFrom<&'a Scalar> for Option<BufferString> {
    type Error = VortexError;

    fn try_from(scalar: &'a Scalar) -> Result<Self, Self::Error> {
        Ok(scalar
            .as_utf8_opt()
            .ok_or_else(|| vortex_err!("Expected utf8 scalar, found {}", scalar.dtype()))?
            .value()
            .cloned())
    }
}

impl TryFrom<Scalar> for Option<BufferString> {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        Self::try_from(&scalar)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// List (`Vec`) conversions.
////////////////////////////////////////////////////////////////////////////////////////////////////

impl<T> TryFrom<Scalar> for Vec<T>
where
    T: for<'b> TryFrom<&'b Scalar, Error = VortexError>,
{
    type Error = VortexError;

    fn try_from(value: Scalar) -> Result<Self, Self::Error> {
        Vec::try_from(&value)
    }
}

impl<'a, T> TryFrom<&'a Scalar> for Vec<T>
where
    T: for<'b> TryFrom<&'b Scalar, Error = VortexError>,
{
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        value
            .as_list_opt()
            .ok_or_else(|| vortex_err!("Expected list scalar, found {}", value.dtype()))?
            .elements()
            .ok_or_else(|| vortex_err!("Expected non-null list"))?
            .into_iter()
            .map(|e| T::try_from(&e))
            .collect::<VortexResult<Vec<T>>>()
    }
}

use std::sync::Arc;

use vortex_buffer::BufferString;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_err};

use crate::ScalarValue;
use crate::scalarvalue::InnerScalarValue;

impl<'a> TryFrom<&'a ScalarValue> for String {
    type Error = VortexError;

    fn try_from(value: &'a ScalarValue) -> Result<Self, Self::Error> {
        Ok(value
            .as_buffer_string()?
            .vortex_expect("Can't convert null ScalarValue to String")
            .to_string())
    }
}

impl From<&str> for ScalarValue {
    fn from(value: &str) -> Self {
        ScalarValue(InnerScalarValue::BufferString(Arc::new(
            value.to_string().into(),
        )))
    }
}

impl From<String> for ScalarValue {
    fn from(value: String) -> Self {
        ScalarValue(InnerScalarValue::BufferString(Arc::new(value.into())))
    }
}

impl From<BufferString> for ScalarValue {
    fn from(value: BufferString) -> Self {
        ScalarValue(InnerScalarValue::BufferString(Arc::new(value)))
    }
}

impl<'a> TryFrom<&'a ScalarValue> for BufferString {
    type Error = VortexError;

    fn try_from(scalar: &'a ScalarValue) -> VortexResult<Self> {
        <Option<BufferString>>::try_from(scalar)?
            .ok_or_else(|| vortex_err!("Can't extract present value from null scalar"))
    }
}

impl<'a> TryFrom<&'a ScalarValue> for Option<BufferString> {
    type Error = VortexError;

    fn try_from(scalar: &'a ScalarValue) -> Result<Self, Self::Error> {
        scalar.as_buffer_string()
    }
}

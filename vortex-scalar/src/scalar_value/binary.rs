use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::{VortexError, VortexExpect, VortexResult};

use crate::ScalarValue;
use crate::scalar_value::InnerScalarValue;

impl<'a> TryFrom<&'a ScalarValue> for ByteBuffer {
    type Error = VortexError;

    fn try_from(scalar: &'a ScalarValue) -> VortexResult<Self> {
        Ok(scalar
            .as_buffer()?
            .vortex_expect("Can't convert null scalar into a byte buffer")
            .as_ref()
            .clone())
    }
}

impl<'a> TryFrom<&'a ScalarValue> for Option<ByteBuffer> {
    type Error = VortexError;

    fn try_from(scalar: &'a ScalarValue) -> VortexResult<Self> {
        Ok(scalar.as_buffer()?.as_ref().map(|b| b.as_ref().clone()))
    }
}

impl From<&[u8]> for ScalarValue {
    fn from(value: &[u8]) -> Self {
        ScalarValue::from(ByteBuffer::from(value.to_vec()))
    }
}

impl From<ByteBuffer> for ScalarValue {
    fn from(value: ByteBuffer) -> Self {
        ScalarValue(InnerScalarValue::Buffer(Arc::new(value)))
    }
}

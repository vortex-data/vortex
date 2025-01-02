use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::str::Utf8Error;

use crate::ByteBuffer;

/// A wrapper around a [`ByteBuffer`] that guarantees that the buffer contains valid UTF-8.
#[derive(Clone, PartialEq, Eq, PartialOrd)]
pub struct BufferString(ByteBuffer);

impl BufferString {
    /// Creates a new `BufferString` from a [`ByteBuffer`].
    ///
    /// # Safety
    /// Assumes that the buffer contains valid UTF-8.
    pub const unsafe fn new_unchecked(buffer: ByteBuffer) -> Self {
        Self(buffer)
    }

    /// Return a view of the contents of BufferString as an immutable `&str`.
    pub fn as_str(&self) -> &str {
        // SAFETY: We have already validated that the buffer is valid UTF-8
        unsafe { std::str::from_utf8_unchecked(self.0.as_ref()) }
    }

    /// Returns the inner [`ByteBuffer`].
    pub fn into_inner(self) -> ByteBuffer {
        self.0
    }
}

impl Debug for BufferString {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferString")
            .field("string", &self.as_str())
            .finish()
    }
}

impl From<BufferString> for ByteBuffer {
    fn from(value: BufferString) -> Self {
        value.0
    }
}

impl From<String> for BufferString {
    fn from(value: String) -> Self {
        Self(ByteBuffer::from(value.into_bytes()))
    }
}

impl From<&str> for BufferString {
    fn from(value: &str) -> Self {
        Self(ByteBuffer::from(String::from(value).into_bytes()))
    }
}

impl TryFrom<ByteBuffer> for BufferString {
    type Error = Utf8Error;

    fn try_from(value: ByteBuffer) -> Result<Self, Self::Error> {
        let _ = std::str::from_utf8(value.as_ref())?;
        Ok(Self(value))
    }
}

impl Deref for BufferString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl AsRef<str> for BufferString {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<[u8]> for BufferString {
    fn as_ref(&self) -> &[u8] {
        self.as_str().as_bytes()
    }
}

#[cfg(test)]
mod test {
    use crate::{buffer, Alignment, BufferString};

    #[test]
    fn buffer_string() {
        let buf = BufferString::from("hello");
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.into_inner().alignment(), Alignment::of::<u8>());
    }

    #[test]
    fn buffer_string_non_ut8() {
        assert!(BufferString::try_from(buffer![0u8, 255]).is_err());
    }
}

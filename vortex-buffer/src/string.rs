use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::str::Utf8Error;

use crate::{AlignedBuffer, Buffer};

/// A wrapper around a [`AlignedBuffer`] that guarantees that the buffer contains valid UTF-8.
#[derive(Clone, PartialEq, Eq, PartialOrd)]
pub struct BufferString(AlignedBuffer);

impl BufferString {
    /// Creates a new `BufferString` from a [`Buffer`].
    ///
    /// # Safety
    /// Assumes that the buffer contains valid UTF-8.
    pub const unsafe fn new_unchecked(buffer: AlignedBuffer) -> Self {
        Self(buffer)
    }

    /// Return a view of the contents of BufferString as an immutable `&str`.
    pub fn as_str(&self) -> &str {
        // SAFETY: We have already validated that the buffer is valid UTF-8
        unsafe { std::str::from_utf8_unchecked(self.0.as_ref()) }
    }
}

impl Debug for BufferString {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferString")
            .field("string", &self.as_str())
            .finish()
    }
}

impl From<BufferString> for AlignedBuffer {
    fn from(value: BufferString) -> Self {
        value.0
    }
}

impl From<String> for BufferString {
    fn from(value: String) -> Self {
        Self(AlignedBuffer::from(value.into_bytes()))
    }
}

impl From<&str> for BufferString {
    fn from(value: &str) -> Self {
        Self(AlignedBuffer::from(String::from(value).into_bytes()))
    }
}

impl TryFrom<Buffer> for BufferString {
    type Error = Utf8Error;

    fn try_from(value: Buffer) -> Result<Self, Self::Error> {
        let _ = std::str::from_utf8(value.as_ref())?;
        // FIXME(ngates): for now this copies
        Ok(Self(AlignedBuffer::from(value.as_ref().to_vec())))
    }
}

impl TryFrom<AlignedBuffer> for BufferString {
    type Error = Utf8Error;

    fn try_from(value: AlignedBuffer) -> Result<Self, Self::Error> {
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

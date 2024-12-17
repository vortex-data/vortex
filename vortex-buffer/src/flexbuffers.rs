#![cfg(feature = "flexbuffers")]
use std::ops::Range;
use std::str::Utf8Error;

use crate::string::BufferString;
use crate::Buffer;

impl flexbuffers::Buffer for Buffer {
    type BufferString = BufferString;

    fn slice(&self, range: Range<usize>) -> Option<Self> {
        if range.end > self.len() || range.start >= self.len() || range.start >= range.end {
            return None;
        }
        Some(Buffer::slice_with_length(self, range.start, range.len()))
    }

    fn empty() -> Self {
        Self::empty()
    }

    fn buffer_str(&self) -> Result<Self::BufferString, Utf8Error> {
        BufferString::try_from(self.clone())
    }
}

use std::ops::Range;

use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_flatbuffers::footer as fb;

/// The location of a segment within a Vortex file.
///
/// A segment is a contiguous block of bytes in a file that contains a part of the file's data.
/// The `SegmentSpec` struct specifies the location and properties of a segment.
#[derive(Clone, Debug)]
pub struct SegmentSpec {
    /// The byte offset of the segment from the start of the file.
    pub offset: u64,
    /// The length of the segment in bytes.
    pub length: u32,
    /// The memory alignment requirement of the segment.
    pub alignment: Alignment,
}

impl SegmentSpec {
    /// Returns the byte range of the segment within the file.
    ///
    /// The range starts at the segment's offset and extends for its length.
    pub fn byte_range(&self) -> Range<u64> {
        self.offset..self.offset + u64::from(self.length)
    }
}

impl From<&SegmentSpec> for fb::SegmentSpec {
    fn from(value: &SegmentSpec) -> Self {
        fb::SegmentSpec::new(value.offset, value.length, value.alignment.exponent(), 0, 0)
    }
}

impl TryFrom<&fb::SegmentSpec> for SegmentSpec {
    type Error = VortexError;

    fn try_from(value: &fb::SegmentSpec) -> Result<Self, Self::Error> {
        Ok(Self {
            offset: value.offset(),
            length: value.length(),
            alignment: Alignment::from_exponent(value.alignment_exponent()),
        })
    }
}

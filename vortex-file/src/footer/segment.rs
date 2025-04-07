use std::ops::Range;

use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_flatbuffers::footer as fb;

/// The location of a segment within a Vortex file.
#[derive(Clone, Debug)]
pub struct SegmentSpec {
    pub offset: u64,
    pub length: u32,
    pub alignment: Alignment,
}

impl SegmentSpec {
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

use vortex_buffer::Alignment;
use vortex_error::{vortex_err, VortexError};
use vortex_flatbuffers::footer2 as fb;

/// The location of a segment within a Vortex file.
#[derive(Clone, Debug)]
pub(crate) struct Segment {
    pub(crate) offset: u64,
    pub(crate) length: u32,
    pub(crate) alignment: Alignment,
}

impl From<&Segment> for fb::Segment {
    fn from(value: &Segment) -> Self {
        fb::Segment::new(value.offset, value.length, value.alignment.exponent(), 0, 0)
    }
}

impl TryFrom<&fb::Segment> for Segment {
    type Error = VortexError;

    fn try_from(value: &fb::Segment) -> Result<Self, Self::Error> {
        Ok(Self {
            offset: value.offset(),
            length: u32::try_from(value.length())
                .map_err(|_| vortex_err!("segment length exceeds maximum u32"))?,
            alignment: Alignment::from_exponent(value.alignment_exponent()),
        })
    }
}

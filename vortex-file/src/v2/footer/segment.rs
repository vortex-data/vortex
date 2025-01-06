use flatbuffers::{FlatBufferBuilder, WIPOffset};
use vortex_flatbuffers::{footer2 as fb, WriteFlatBuffer};

/// The location of a segment within a Vortex file.
pub(crate) struct Segment {
    pub(crate) offset: u64,
    pub(crate) length: usize,
}

impl WriteFlatBuffer for Segment {
    type Target<'a> = fb::Segment<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        fb::Segment::create(
            fbb,
            &fb::SegmentArgs {
                offset: self.offset,
                length: self.length as u64,
            },
        )
    }
}

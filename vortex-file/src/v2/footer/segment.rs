use flatbuffers::{FlatBufferBuilder, Follow, WIPOffset};
use vortex_error::{vortex_err, VortexError};
use vortex_flatbuffers::{footer2 as fb, ReadFlatBuffer, WriteFlatBuffer};

/// The location of a segment within a Vortex file.
#[derive(Clone, Debug)]
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

impl ReadFlatBuffer for Segment {
    type Source<'a> = fb::Segment<'a>;
    type Error = VortexError;

    fn read_flatbuffer<'buf>(
        fb: &<Self::Source<'buf> as Follow<'buf>>::Inner,
    ) -> Result<Self, Self::Error> {
        Ok(Self {
            offset: fb.offset(),
            length: usize::try_from(fb.length())
                .map_err(|_| vortex_err!("segment length exceeds maximum usize"))?,
        })
    }
}

use vortex_flatbuffers::{footer2 as fb, FlatBufferRoot, WriteFlatBuffer};

use crate::v2::footer::segment::Segment;

/// Captures the layout information of a Vortex file.
pub(crate) struct Postscript {
    pub(crate) dtype: Segment,
    pub(crate) file_layout: Segment,
}

impl FlatBufferRoot for Postscript {}

impl WriteFlatBuffer for Postscript {
    type Target<'a> = fb::Postscript<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut flatbuffers::FlatBufferBuilder<'fb>,
    ) -> flatbuffers::WIPOffset<Self::Target<'fb>> {
        let dtype = self.dtype.write_flatbuffer(fbb);
        let file_layout = self.file_layout.write_flatbuffer(fbb);
        fb::Postscript::create(
            fbb,
            &fb::PostscriptArgs {
                dtype: Some(dtype),
                file_layout: Some(file_layout),
            },
        )
    }
}

use flatbuffers::Follow;
use vortex_error::{vortex_err, VortexError};
use vortex_flatbuffers::{footer2 as fb, FlatBufferRoot, ReadFlatBuffer, WriteFlatBuffer};

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

impl ReadFlatBuffer for Postscript {
    type Source<'a> = fb::Postscript<'a>;
    type Error = VortexError;

    fn read_flatbuffer<'buf>(
        fb: &<Self::Source<'buf> as Follow<'buf>>::Inner,
    ) -> Result<Self, Self::Error> {
        Ok(Self {
            dtype: Segment::read_flatbuffer(
                &fb.dtype()
                    .ok_or_else(|| vortex_err!("Postscript missing dtype segment"))?,
            )?,
            file_layout: Segment::read_flatbuffer(
                &fb.file_layout()
                    .ok_or_else(|| vortex_err!("Postscript missing file_layout segment"))?,
            )?,
        })
    }
}

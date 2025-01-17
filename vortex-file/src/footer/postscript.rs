use flatbuffers::Follow;
use vortex_error::{vortex_err, VortexError};
use vortex_flatbuffers::{footer2 as fb, FlatBufferRoot, ReadFlatBuffer, WriteFlatBuffer};

use crate::footer::segment::Segment;

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
        let dtype = fb::Segment::from(&self.dtype);
        let file_layout = fb::Segment::from(&self.file_layout);
        fb::Postscript::create(
            fbb,
            &fb::PostscriptArgs {
                dtype: Some(&dtype),
                file_layout: Some(&file_layout),
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
            dtype: Segment::try_from(
                fb.dtype()
                    .ok_or_else(|| vortex_err!("Postscript missing dtype segment"))?,
            )?,
            file_layout: Segment::try_from(
                fb.file_layout()
                    .ok_or_else(|| vortex_err!("Postscript missing file_layout segment"))?,
            )?,
        })
    }
}

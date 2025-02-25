use flatbuffers::Follow;
use vortex_error::{vortex_err, VortexError};
use vortex_flatbuffers::{footer as fb, FlatBufferRoot, ReadFlatBuffer, WriteFlatBuffer};

use crate::footer::segment::Segment;

/// Captures the layout information of a Vortex file.
pub(crate) struct Postscript {
    pub(crate) dtype: Option<Segment>,
    pub(crate) footer: Segment,
}

impl FlatBufferRoot for Postscript {}

impl WriteFlatBuffer for Postscript {
    type Target<'a> = fb::Postscript<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut flatbuffers::FlatBufferBuilder<'fb>,
    ) -> flatbuffers::WIPOffset<Self::Target<'fb>> {
        let dtype = self.dtype.as_ref().map(fb::Segment::from);
        let footer = fb::Segment::from(&self.footer);
        fb::Postscript::create(
            fbb,
            &fb::PostscriptArgs {
                dtype: dtype.as_ref(),
                footer: Some(&footer),
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
            dtype: fb.dtype().map(Segment::try_from).transpose()?,
            footer: Segment::try_from(
                fb.footer()
                    .ok_or_else(|| vortex_err!("Postscript missing footer segment"))?,
            )?,
        })
    }
}

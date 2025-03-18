use flatbuffers::Follow;
use vortex_error::{VortexError, vortex_err};
use vortex_flatbuffers::{FlatBufferRoot, ReadFlatBuffer, WriteFlatBuffer, footer as fb};

use crate::footer::segment::SegmentSpec;

/// Captures the layout information of a Vortex file.
pub(crate) struct Postscript {
    pub(crate) dtype: Option<SegmentSpec>,
    pub(crate) footer: SegmentSpec,
}

impl FlatBufferRoot for Postscript {}

impl WriteFlatBuffer for Postscript {
    type Target<'a> = fb::Postscript<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut flatbuffers::FlatBufferBuilder<'fb>,
    ) -> flatbuffers::WIPOffset<Self::Target<'fb>> {
        let dtype = self.dtype.as_ref().map(fb::SegmentSpec::from);
        let footer = fb::SegmentSpec::from(&self.footer);
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
            dtype: fb.dtype().map(SegmentSpec::try_from).transpose()?,
            footer: SegmentSpec::try_from(
                fb.footer()
                    .ok_or_else(|| vortex_err!("Postscript missing footer segment"))?,
            )?,
        })
    }
}

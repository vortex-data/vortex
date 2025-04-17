use flatbuffers::{FlatBufferBuilder, Follow, WIPOffset};
use vortex_buffer::Alignment;
use vortex_error::{VortexError, vortex_err};
use vortex_flatbuffers::{FlatBufferRoot, ReadFlatBuffer, WriteFlatBuffer, footer as fb};

/// The postscript captures the locations and compression for the initial segments required for
/// reading a Vortex file.
pub(crate) struct Postscript {
    pub(crate) dtype: Option<PostscriptSegment>,
    pub(crate) statistics: Option<PostscriptSegment>,
    pub(crate) layout: PostscriptSegment,
}

impl FlatBufferRoot for Postscript {}

impl WriteFlatBuffer for Postscript {
    type Target<'a> = fb::Postscript<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let dtype = self.dtype.as_ref().map(|ps| ps.write_flatbuffer(fbb));
        let statistics = self.statistics.as_ref().map(|ps| ps.write_flatbuffer(fbb));
        let layout = self.layout.write_flatbuffer(fbb);
        fb::Postscript::create(
            fbb,
            &fb::PostscriptArgs {
                dtype,
                statistics,
                layout: Some(layout),
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
            dtype: fb
                .dtype()
                .map(|ps| PostscriptSegment::read_flatbuffer(&ps))
                .transpose()?,
            statistics: fb
                .statistics()
                .map(|ps| PostscriptSegment::read_flatbuffer(&ps))
                .transpose()?,
            layout: PostscriptSegment::read_flatbuffer(
                &fb.layout()
                    .ok_or_else(|| vortex_err!("Postscript missing layout segment"))?,
            )?,
        })
    }
}

pub struct PostscriptSegment {
    pub(crate) offset: u64,
    pub(crate) length: u32,
    pub(crate) alignment: Alignment,
}

impl FlatBufferRoot for PostscriptSegment {}

impl WriteFlatBuffer for PostscriptSegment {
    type Target<'a> = fb::PostscriptSegment<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        fb::PostscriptSegment::create(
            fbb,
            &fb::PostscriptSegmentArgs {
                offset: self.offset,
                length: self.length,
                alignment_exponent: self.alignment.exponent(),
                _compression: None,
                _encryption: None,
            },
        )
    }
}

impl ReadFlatBuffer for PostscriptSegment {
    type Source<'a> = fb::PostscriptSegment<'a>;
    type Error = VortexError;

    fn read_flatbuffer<'buf>(
        fb: &<Self::Source<'buf> as Follow<'buf>>::Inner,
    ) -> Result<Self, Self::Error> {
        Ok(PostscriptSegment {
            offset: fb.offset(),
            length: fb.length(),
            alignment: Alignment::from_exponent(fb.alignment_exponent()),
        })
    }
}

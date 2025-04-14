use flatbuffers::{FlatBufferBuilder, Follow, WIPOffset};
use vortex_array::Array;
use vortex_error::{VortexError, vortex_err};
use vortex_flatbuffers::{FlatBufferRoot, ReadFlatBuffer, WriteFlatBuffer, footer as fb};

/// Captures the layout information of a Vortex file.
pub(crate) struct Postscript {
    pub(crate) dtype: Option<PostscriptSegment>,
    pub(crate) statistics: Option<PostscriptSegment>,
    pub(crate) layout: PostscriptSegment,
    pub(crate) registry: PostscriptSegment,
}

impl FlatBufferRoot for Postscript {}

impl WriteFlatBuffer for Postscript {
    type Target<'a> = fb::Postscript<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut flatbuffers::FlatBufferBuilder<'fb>,
    ) -> flatbuffers::WIPOffset<Self::Target<'fb>> {
        let dtype = self.dtype.as_ref().map(fb::SegmentSpec::from);
        let statistics = self.statistics.as_ref().map(fb::SegmentSpec::from);
        let layout = fb::PostscriptSegment::from(&self.layout);
        let registry = fb::SegmentSpec::from(&self.registry);
        fb::Postscript::create(
            fbb,
            &fb::PostscriptArgs {
                dtype: dtype.as_ref(),
                layout: Some(layout),
                statistics: statistics.as_ref(),
                registry: Some(registry),
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
            dtype: fb.dtype().map(PostscriptSegment::try_from).transpose()?,
            statistics: fb
                .statistics()
                .map(PostscriptSegment::try_from)
                .transpose()?,
            layout: PostscriptSegment::try_from(
                fb.layout()
                    .ok_or_else(|| vortex_err!("Postscript missing layout segment"))?,
            )?,
            registry: PostscriptSegment::try_from(
                fb.registry()
                    .ok_or_else(|| vortex_err!("Postscript missing registry segment"))?,
            )?,
        })
    }
}

pub struct PostscriptSegment {
    pub(crate) offset: u64,
    pub(crate) length: u32,
    pub(crate) alignment_exponent: u8,
}

impl FlatBufferRoot for PostscriptSegment {}

impl WriteFlatBuffer for PostscriptSegment {
    type Target<'a> = fb::PostscriptSegment<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        todo!()
    }
}

impl ReadFlatBuffer for PostscriptSegment {
    type Source<'a> = fb::PostscriptSegment<'a>;
    type Error = VortexError;

    fn read_flatbuffer<'buf>(
        fb: &<Self::Source<'buf> as Follow<'buf>>::Inner,
    ) -> Result<Self, Self::Error> {
        todo!()
    }
}

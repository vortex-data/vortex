// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBufferBuilder;
use vortex_flatbuffers::FlatBufferRoot;
use vortex_flatbuffers::ReadFlatBuffer;
use vortex_flatbuffers::WIPOffset;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::footer as fb;

/// The postscript captures the locations and compression for the initial segments required for
/// reading a Vortex file.
pub(crate) struct Postscript {
    pub(crate) dtype: Option<PostscriptSegment>,
    pub(crate) layout: PostscriptSegment,
    pub(crate) statistics: Option<PostscriptSegment>,
    pub(crate) footer: PostscriptSegment,
}

impl FlatBufferRoot for Postscript {}

impl WriteFlatBuffer for Postscript {
    type Target = fb::Postscript;

    fn write_flatbuffer(
        &self,
        fbb: &mut FlatBufferBuilder,
    ) -> VortexResult<WIPOffset<Self::Target>> {
        let dtype = self
            .dtype
            .as_ref()
            .map(|ps| ps.write_flatbuffer(fbb))
            .transpose()?;
        let layout = self.layout.write_flatbuffer(fbb)?;
        let statistics = self
            .statistics
            .as_ref()
            .map(|ps| ps.write_flatbuffer(fbb))
            .transpose()?;
        let footer = self.footer.write_flatbuffer(fbb)?;
        Ok(fb::Postscript::create(
            fbb,
            dtype,
            Some(layout),
            statistics,
            Some(footer),
        ))
    }
}

impl ReadFlatBuffer for Postscript {
    type Source<'a> = fb::PostscriptRef<'a>;
    type Error = VortexError;

    fn read_flatbuffer<'buf>(fb: &Self::Source<'buf>) -> Result<Self, Self::Error> {
        Ok(Self {
            dtype: fb
                .dtype()?
                .map(|ps| PostscriptSegment::read_flatbuffer(&ps))
                .transpose()?,
            layout: PostscriptSegment::read_flatbuffer(
                &fb.layout()?
                    .ok_or_else(|| vortex_err!("Postscript missing layout segment"))?,
            )?,
            statistics: fb
                .statistics()?
                .map(|ps| PostscriptSegment::read_flatbuffer(&ps))
                .transpose()?,
            footer: PostscriptSegment::read_flatbuffer(
                &fb.footer()?
                    .ok_or_else(|| vortex_err!("Postscript missing footer segment"))?,
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
    type Target = fb::PostscriptSegment;

    fn write_flatbuffer(
        &self,
        fbb: &mut FlatBufferBuilder,
    ) -> VortexResult<WIPOffset<Self::Target>> {
        Ok(fb::PostscriptSegment::create(
            fbb,
            self.offset,
            self.length,
            self.alignment.exponent(),
            None::<fb::CompressionSpec>,
            None::<fb::EncryptionSpec>,
        ))
    }
}

impl ReadFlatBuffer for PostscriptSegment {
    type Source<'a> = fb::PostscriptSegmentRef<'a>;
    type Error = VortexError;

    fn read_flatbuffer<'buf>(fb: &Self::Source<'buf>) -> Result<Self, Self::Error> {
        Ok(PostscriptSegment {
            offset: fb.offset()?,
            length: fb.length()?,
            alignment: Alignment::from_exponent(fb.alignment_exponent()?),
        })
    }
}

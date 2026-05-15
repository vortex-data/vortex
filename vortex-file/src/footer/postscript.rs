// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use flatbuffers::FlatBufferBuilder;
use flatbuffers::Follow;
use flatbuffers::WIPOffset;
use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBufferRoot;
use vortex_flatbuffers::ReadFlatBuffer;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::footer as fb;
use vortex_utils::aliases::hash_set::HashSet;

/// The postscript captures the locations and compression for the initial segments required for
/// reading a Vortex file.
pub(crate) struct Postscript {
    pub(crate) dtype: Option<PostscriptSegment>,
    pub(crate) layout: PostscriptSegment,
    pub(crate) statistics: Option<PostscriptSegment>,
    pub(crate) footer: PostscriptSegment,
    pub(crate) metadata: Vec<PostscriptMetadata>,
}

impl FlatBufferRoot for Postscript {}

impl WriteFlatBuffer for Postscript {
    type Target<'a> = fb::Postscript<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
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
        let metadata = if self.metadata.is_empty() {
            None
        } else {
            let mut metadata = Vec::with_capacity(self.metadata.len());
            for entry in &self.metadata {
                metadata.push(entry.write_flatbuffer(fbb)?);
            }
            Some(fbb.create_vector(metadata.as_slice()))
        };
        Ok(fb::Postscript::create(
            fbb,
            &fb::PostscriptArgs {
                dtype,
                layout: Some(layout),
                statistics,
                footer: Some(footer),
                metadata,
            },
        ))
    }
}

impl ReadFlatBuffer for Postscript {
    type Source<'a> = fb::Postscript<'a>;
    type Error = VortexError;

    fn read_flatbuffer<'buf>(
        fb: &<Self::Source<'buf> as Follow<'buf>>::Inner,
    ) -> Result<Self, Self::Error> {
        let metadata = fb
            .metadata()
            .map(|metadata| {
                metadata
                    .iter()
                    .map(|entry| PostscriptMetadata::read_flatbuffer(&entry))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();

        {
            let mut seen_keys = HashSet::with_capacity(metadata.len());
            for entry in &metadata {
                if entry.key.is_empty() {
                    return Err(vortex_err!("Postscript metadata key must not be empty"));
                }
                if !seen_keys.insert(&entry.key) {
                    return Err(vortex_err!(
                        "Postscript contains duplicate metadata key {}",
                        entry.key
                    ));
                }
            }
        }

        Ok(Self {
            dtype: fb
                .dtype()
                .map(|ps| PostscriptSegment::read_flatbuffer(&ps))
                .transpose()?,
            layout: PostscriptSegment::read_flatbuffer(
                &fb.layout()
                    .ok_or_else(|| vortex_err!("Postscript missing layout segment"))?,
            )?,
            statistics: fb
                .statistics()
                .map(|ps| PostscriptSegment::read_flatbuffer(&ps))
                .transpose()?,
            footer: PostscriptSegment::read_flatbuffer(
                &fb.footer()
                    .ok_or_else(|| vortex_err!("Postscript missing footer segment"))?,
            )?,
            metadata,
        })
    }
}

pub(crate) struct PostscriptMetadata {
    pub(crate) key: String,
    pub(crate) segment: PostscriptSegment,
}

impl FlatBufferRoot for PostscriptMetadata {}

impl WriteFlatBuffer for PostscriptMetadata {
    type Target<'a> = fb::PostscriptMetadata<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
        let key = fbb.create_string(&self.key);
        let segment = self.segment.write_flatbuffer(fbb)?;
        Ok(fb::PostscriptMetadata::create(
            fbb,
            &fb::PostscriptMetadataArgs {
                key: Some(key),
                segment: Some(segment),
            },
        ))
    }
}

impl ReadFlatBuffer for PostscriptMetadata {
    type Source<'a> = fb::PostscriptMetadata<'a>;
    type Error = VortexError;

    fn read_flatbuffer<'buf>(
        fb: &<Self::Source<'buf> as Follow<'buf>>::Inner,
    ) -> Result<Self, Self::Error> {
        // The FlatBuffers verifier enforces that `key` and `segment` are present before this
        // accessor can panic on a malformed required field.
        Ok(Self {
            key: fb.key().to_string(),
            segment: PostscriptSegment::read_flatbuffer(&fb.segment())?,
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
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
        Ok(fb::PostscriptSegment::create(
            fbb,
            &fb::PostscriptSegmentArgs {
                offset: self.offset,
                length: self.length,
                alignment_exponent: self.alignment.exponent(),
                _compression: None,
                _encryption: None,
            },
        ))
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

#[cfg(test)]
mod tests {
    use vortex_flatbuffers::ReadFlatBuffer;
    use vortex_flatbuffers::WriteFlatBufferExt;

    use super::*;

    fn segment(offset: u64) -> PostscriptSegment {
        PostscriptSegment {
            offset,
            length: 1,
            alignment: Alignment::none(),
        }
    }

    fn read_postscript_error(postscript: &Postscript) -> String {
        let bytes = postscript.write_flatbuffer_bytes().unwrap();
        match Postscript::read_flatbuffer_bytes(&bytes) {
            Ok(_) => panic!("expected postscript read to fail"),
            Err(err) => err.to_string(),
        }
    }

    #[test]
    fn duplicate_metadata_keys_are_rejected() {
        let err = read_postscript_error(&Postscript {
            dtype: None,
            layout: segment(0),
            statistics: None,
            footer: segment(1),
            metadata: vec![
                PostscriptMetadata {
                    key: "metadata".to_string(),
                    segment: segment(2),
                },
                PostscriptMetadata {
                    key: "metadata".to_string(),
                    segment: segment(3),
                },
            ],
        });

        assert!(err.contains("duplicate metadata key"));
    }

    #[test]
    fn empty_metadata_keys_are_rejected() {
        let err = read_postscript_error(&Postscript {
            dtype: None,
            layout: segment(0),
            statistics: None,
            footer: segment(1),
            metadata: vec![PostscriptMetadata {
                key: String::new(),
                segment: segment(2),
            }],
        });

        assert!(err.contains("metadata key must not be empty"));
    }
}

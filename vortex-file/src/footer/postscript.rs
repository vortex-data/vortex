// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use flatbuffers::FlatBufferBuilder;
use flatbuffers::Follow;
use flatbuffers::WIPOffset;
use vortex_array::flatbuffers::FlatBufferRoot;
use vortex_array::flatbuffers::ReadFlatBuffer;
use vortex_array::flatbuffers::WriteFlatBuffer;
use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_set::HashSet;

use super::MAX_METADATA_KEY_BYTES;
use super::MAX_METADATA_SEGMENTS;
use crate::encryption::EncryptionSpec;
use crate::flatbuffers::footer as fb;

/// The postscript captures the locations and compression for the initial segments required for
/// reading a Vortex file.
pub(crate) struct Postscript {
    pub(crate) dtype: Option<PostscriptSegment>,
    pub(crate) layout: PostscriptSegment,
    pub(crate) statistics: Option<PostscriptSegment>,
    pub(crate) footer: PostscriptSegment,
    pub(crate) metadata: Vec<PostscriptMetadata>,
    pub(crate) encrypted_statistics: Vec<EncryptedStatistics>,
}

/// File-level statistics of some fields, stored in a segment encrypted with those fields' key.
pub(crate) struct EncryptedStatistics {
    pub(crate) fields: Vec<u32>,
    pub(crate) segment: PostscriptSegment,
}

impl Postscript {
    /// The [`postscript_binding`] of this postscript.
    pub(crate) fn binding(&self) -> Vec<u8> {
        postscript_binding(
            self.dtype.is_some(),
            self.statistics.is_some(),
            self.metadata.iter().map(|entry| entry.key.as_str()),
            self.encrypted_statistics
                .iter()
                .map(|entry| entry.fields.as_slice()),
        )
    }
}

/// Encode the postscript entries that a protected footer authenticates through its additional
/// authenticated data: whether the dtype and statistics segments are present, the keys of the
/// metadata segments, and the fields of the encrypted statistics segments, in order.
///
/// The postscript itself is not authenticated, so without this an attacker could drop these
/// entries unnoticed.
pub(crate) fn postscript_binding<'a>(
    has_dtype: bool,
    has_statistics: bool,
    metadata_keys: impl Iterator<Item = &'a str>,
    encrypted_statistics: impl Iterator<Item = &'a [u32]>,
) -> Vec<u8> {
    let mut binding = vec![u8::from(has_dtype) | (u8::from(has_statistics) << 1)];
    let mut count = 0u64;
    let mut keys = Vec::new();
    for key in metadata_keys {
        count += 1;
        keys.extend_from_slice(&(key.len() as u64).to_le_bytes());
        keys.extend_from_slice(key.as_bytes());
    }
    binding.extend_from_slice(&count.to_le_bytes());
    binding.extend_from_slice(&keys);

    let mut count = 0u64;
    let mut entries = Vec::new();
    for fields in encrypted_statistics {
        count += 1;
        entries.extend_from_slice(&(fields.len() as u64).to_le_bytes());
        for field in fields {
            entries.extend_from_slice(&field.to_le_bytes());
        }
    }
    binding.extend_from_slice(&count.to_le_bytes());
    binding.extend_from_slice(&entries);
    binding
}

/// The `EncryptionSpec` tables already written to a postscript, by spec.
#[derive(Default)]
struct SharedSpecs<'fb>(Vec<(EncryptionSpec, WIPOffset<fb::EncryptionSpec<'fb>>)>);

impl<'fb> SharedSpecs<'fb> {
    fn offset(
        &mut self,
        fbb: &mut FlatBufferBuilder<'fb>,
        spec: &EncryptionSpec,
    ) -> WIPOffset<fb::EncryptionSpec<'fb>> {
        if let Some((_, offset)) = self.0.iter().find(|(known, _)| known == spec) {
            return *offset;
        }
        let offset = spec.write_flatbuffer(fbb);
        self.0.push((spec.clone(), offset));
        offset
    }
}

impl EncryptedStatistics {
    fn write_flatbuffer_shared<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
        specs: &mut SharedSpecs<'fb>,
    ) -> VortexResult<WIPOffset<fb::EncryptedStatistics<'fb>>> {
        let fields = fbb.create_vector(&self.fields);
        let segment = self.segment.write_flatbuffer_shared(fbb, specs)?;
        Ok(fb::EncryptedStatistics::create(
            fbb,
            &fb::EncryptedStatisticsArgs {
                fields: Some(fields),
                segment: Some(segment),
            },
        ))
    }

    fn read_flatbuffer(fb: &fb::EncryptedStatistics<'_>) -> VortexResult<Self> {
        // The FlatBuffers verifier enforces that `fields` and `segment` are present.
        Ok(Self {
            fields: fb.fields().iter().collect(),
            segment: PostscriptSegment::read_flatbuffer(&fb.segment())?,
        })
    }
}

impl FlatBufferRoot for Postscript {}

impl WriteFlatBuffer for Postscript {
    type Target<'a> = fb::Postscript<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
        validate_metadata_entries(&self.metadata)?;
        self.write_flatbuffer_unchecked(fbb)
    }
}

impl Postscript {
    fn write_flatbuffer_unchecked<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<fb::Postscript<'fb>>> {
        // Segments protected by the same key share one `EncryptionSpec` table, which the
        // segments reference by offset, so key metadata is stored once per key.
        let mut specs = SharedSpecs::default();
        let dtype = self
            .dtype
            .as_ref()
            .map(|ps| ps.write_flatbuffer_shared(fbb, &mut specs))
            .transpose()?;
        let layout = self.layout.write_flatbuffer_shared(fbb, &mut specs)?;
        let statistics = self
            .statistics
            .as_ref()
            .map(|ps| ps.write_flatbuffer_shared(fbb, &mut specs))
            .transpose()?;
        let footer = self.footer.write_flatbuffer_shared(fbb, &mut specs)?;
        let metadata = if self.metadata.is_empty() {
            None
        } else {
            let mut metadata = Vec::with_capacity(self.metadata.len());
            for entry in &self.metadata {
                metadata.push(entry.write_flatbuffer_shared(fbb, &mut specs)?);
            }
            Some(fbb.create_vector(metadata.as_slice()))
        };
        let encrypted_statistics = if self.encrypted_statistics.is_empty() {
            None
        } else {
            let entries = self
                .encrypted_statistics
                .iter()
                .map(|entry| entry.write_flatbuffer_shared(fbb, &mut specs))
                .collect::<VortexResult<Vec<_>>>()?;
            Some(fbb.create_vector(entries.as_slice()))
        };
        Ok(fb::Postscript::create(
            fbb,
            &fb::PostscriptArgs {
                dtype,
                layout: Some(layout),
                statistics,
                footer: Some(footer),
                metadata,
                encrypted_statistics,
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
        let metadata = match fb.metadata() {
            Some(metadata) => {
                let metadata_len = metadata.len();
                if metadata_len > MAX_METADATA_SEGMENTS {
                    return Err(metadata_count_error(metadata_len));
                }

                let mut seen_keys = HashSet::with_capacity(metadata_len);
                let mut entries = Vec::with_capacity(metadata_len);
                for entry in metadata.iter() {
                    let entry = PostscriptMetadata::read_flatbuffer(&entry)?;
                    validate_metadata_key(&entry.key)?;
                    if !seen_keys.insert(entry.key.clone()) {
                        return Err(duplicate_metadata_key_error(&entry.key));
                    }
                    entries.push(entry);
                }
                entries
            }
            None => Vec::new(),
        };

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
            encrypted_statistics: fb
                .encrypted_statistics()
                .iter()
                .flat_map(|entries| entries.iter())
                .map(|entry| EncryptedStatistics::read_flatbuffer(&entry))
                .collect::<VortexResult<_>>()?,
        })
    }
}

fn validate_metadata_entries(metadata: &[PostscriptMetadata]) -> VortexResult<()> {
    if metadata.len() > MAX_METADATA_SEGMENTS {
        return Err(metadata_count_error(metadata.len()));
    }

    let mut seen_keys = HashSet::with_capacity(metadata.len());
    for entry in metadata {
        validate_metadata_key(&entry.key)?;
        if !seen_keys.insert(&entry.key) {
            return Err(duplicate_metadata_key_error(&entry.key));
        }
    }

    Ok(())
}

fn metadata_count_error(metadata_len: usize) -> VortexError {
    vortex_err!(
        "Postscript contains {} metadata segments, but Vortex supports at most {} metadata segments; metadata keys must be non-empty and at most {} bytes",
        metadata_len,
        MAX_METADATA_SEGMENTS,
        MAX_METADATA_KEY_BYTES
    )
}

fn duplicate_metadata_key_error(key: &str) -> VortexError {
    vortex_err!(
        "Postscript contains duplicate metadata key {key}; metadata keys must be unique, non-empty, and at most {} bytes, and files may contain at most {} metadata segments",
        MAX_METADATA_KEY_BYTES,
        MAX_METADATA_SEGMENTS
    )
}

fn validate_metadata_key(key: &str) -> VortexResult<()> {
    if key.is_empty() {
        return Err(vortex_err!(
            "Postscript metadata key must not be empty; metadata keys must be at most {} bytes and files may contain at most {} metadata segments",
            MAX_METADATA_KEY_BYTES,
            MAX_METADATA_SEGMENTS
        ));
    }

    let key_bytes = key.len();
    if key_bytes > MAX_METADATA_KEY_BYTES {
        return Err(vortex_err!(
            "Postscript metadata key {key:?} is {key_bytes} bytes, but metadata keys must be at most {} bytes and files may contain at most {} metadata segments",
            MAX_METADATA_KEY_BYTES,
            MAX_METADATA_SEGMENTS
        ));
    }

    Ok(())
}

pub(crate) struct PostscriptMetadata {
    pub(crate) key: String,
    pub(crate) segment: PostscriptSegment,
}

impl PostscriptMetadata {
    fn write_flatbuffer_shared<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
        specs: &mut SharedSpecs<'fb>,
    ) -> VortexResult<WIPOffset<fb::PostscriptMetadata<'fb>>> {
        let key = fbb.create_string(&self.key);
        let segment = self.segment.write_flatbuffer_shared(fbb, specs)?;
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
    pub(crate) encryption: EncryptionSpec,
}

impl PostscriptSegment {
    fn write_flatbuffer_shared<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
        specs: &mut SharedSpecs<'fb>,
    ) -> VortexResult<WIPOffset<fb::PostscriptSegment<'fb>>> {
        let encryption = self
            .encryption
            .is_encrypted()
            .then(|| specs.offset(fbb, &self.encryption));
        Ok(fb::PostscriptSegment::create(
            fbb,
            &fb::PostscriptSegmentArgs {
                offset: self.offset,
                length: self.length,
                alignment_exponent: self.alignment.exponent(),
                _compression: None,
                encryption,
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
            // The alignment exponent comes from the file and may be corrupt, so validate it rather
            // than panicking on a too-large shift (see issue #8819).
            alignment: Alignment::try_from_untrusted_exponent(fb.alignment_exponent())?,
            encryption: fb
                .encryption()
                .map(|spec| EncryptionSpec::read_flatbuffer(&spec))
                .transpose()?
                .unwrap_or_else(EncryptionSpec::plaintext),
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::flatbuffers::FlatBuffer;
    use vortex_array::flatbuffers::ReadFlatBuffer;
    use vortex_array::flatbuffers::WriteFlatBufferExt;
    use vortex_buffer::ByteBuffer;

    use super::*;
    use crate::MAX_POSTSCRIPT_SIZE;

    fn segment(offset: u64) -> PostscriptSegment {
        PostscriptSegment {
            offset,
            length: 1,
            alignment: Alignment::none(),
            encryption: EncryptionSpec::plaintext(),
        }
    }

    fn write_postscript_bytes_unchecked(postscript: &Postscript) -> FlatBuffer {
        let mut fbb = FlatBufferBuilder::new();
        let root_offset = postscript.write_flatbuffer_unchecked(&mut fbb).unwrap();
        fbb.finish_minimal(root_offset);
        let (vec, start) = fbb.collapse();
        let end = vec.len();
        FlatBuffer::align_from(ByteBuffer::from(vec).slice(start..end))
    }

    fn read_postscript_error(postscript: &Postscript) -> String {
        let bytes = write_postscript_bytes_unchecked(postscript);
        match Postscript::read_flatbuffer_bytes(&bytes) {
            Ok(_) => panic!("expected postscript read to fail"),
            Err(err) => err.to_string(),
        }
    }

    fn write_postscript_error(postscript: &Postscript) -> String {
        match postscript.write_flatbuffer_bytes() {
            Ok(_) => panic!("expected postscript write to fail"),
            Err(err) => err.to_string(),
        }
    }

    fn key_limit_message() -> String {
        format!("at most {MAX_METADATA_KEY_BYTES} bytes")
    }

    fn segment_limit_message() -> String {
        format!("at most {MAX_METADATA_SEGMENTS} metadata segments")
    }

    #[test]
    fn metadata_limit_boundaries_roundtrip() {
        let metadata = (0..MAX_METADATA_SEGMENTS)
            .map(|idx| PostscriptMetadata {
                key: format!("{idx:0>width$}", width = MAX_METADATA_KEY_BYTES),
                segment: segment(2 + idx as u64),
            })
            .collect::<Vec<_>>();

        let bytes = Postscript {
            dtype: None,
            layout: segment(0),
            statistics: None,
            footer: segment(1),
            metadata,
            encrypted_statistics: Vec::new(),
        }
        .write_flatbuffer_bytes()
        .unwrap();

        let postscript = Postscript::read_flatbuffer_bytes(&bytes).unwrap();
        assert_eq!(postscript.metadata.len(), MAX_METADATA_SEGMENTS);
        let mut total_key_bytes = 0;
        for (index, entry) in postscript.metadata.iter().enumerate() {
            assert_eq!(
                entry.key,
                format!("{index:0>width$}", width = MAX_METADATA_KEY_BYTES)
            );
            assert_eq!(entry.segment.offset, 2 + index as u64);
            assert_eq!(entry.segment.length, 1);
            assert_eq!(entry.segment.alignment, Alignment::none());
            assert_eq!(entry.key.len(), MAX_METADATA_KEY_BYTES);
            total_key_bytes += entry.key.len();
        }
        assert_eq!(
            total_key_bytes,
            MAX_METADATA_SEGMENTS * MAX_METADATA_KEY_BYTES
        );
        assert!(bytes.len() <= MAX_POSTSCRIPT_SIZE as usize);
    }

    #[test]
    fn duplicate_metadata_keys_are_rejected() {
        let postscript = Postscript {
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
            encrypted_statistics: Vec::new(),
        };

        let read_err = read_postscript_error(&postscript);
        assert!(read_err.contains("duplicate metadata key"));
        assert!(read_err.contains(&key_limit_message()));
        assert!(read_err.contains(&segment_limit_message()));
        let write_err = write_postscript_error(&postscript);
        assert!(write_err.contains("duplicate metadata key"));
        assert!(write_err.contains(&key_limit_message()));
        assert!(write_err.contains(&segment_limit_message()));
    }

    #[test]
    fn empty_metadata_keys_are_rejected() {
        let postscript = Postscript {
            dtype: None,
            layout: segment(0),
            statistics: None,
            footer: segment(1),
            metadata: vec![PostscriptMetadata {
                key: String::new(),
                segment: segment(2),
            }],
            encrypted_statistics: Vec::new(),
        };

        let read_err = read_postscript_error(&postscript);
        assert!(read_err.contains("metadata key must not be empty"));
        let write_err = write_postscript_error(&postscript);
        assert!(write_err.contains("metadata key must not be empty"));
    }

    #[test]
    fn long_metadata_keys_are_rejected() {
        let key = "é".repeat((MAX_METADATA_KEY_BYTES / "é".len()) + 1);
        let key_len = key.len();
        let postscript = Postscript {
            dtype: None,
            layout: segment(0),
            statistics: None,
            footer: segment(1),
            metadata: vec![PostscriptMetadata {
                key,
                segment: segment(2),
            }],
            encrypted_statistics: Vec::new(),
        };

        for err in [
            read_postscript_error(&postscript),
            write_postscript_error(&postscript),
        ] {
            assert!(err.contains(&format!("{key_len} bytes")));
            assert!(err.contains(&key_limit_message()));
            assert!(err.contains(&segment_limit_message()));
        }
    }

    #[test]
    fn too_many_metadata_segments_are_rejected() {
        let postscript = Postscript {
            dtype: None,
            layout: segment(0),
            statistics: None,
            footer: segment(1),
            metadata: (0..=MAX_METADATA_SEGMENTS)
                .map(|idx| PostscriptMetadata {
                    key: format!("metadata-{idx}"),
                    segment: segment(2 + idx as u64),
                })
                .collect(),
            encrypted_statistics: Vec::new(),
        };

        for err in [
            read_postscript_error(&postscript),
            write_postscript_error(&postscript),
        ] {
            assert!(err.contains(&segment_limit_message()));
            assert!(err.contains(&key_limit_message()));
        }
    }
}

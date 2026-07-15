// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! User-defined file-level metadata: opaque byte values keyed by unique, non-empty strings,
//! serialized into a single [`fb::FileMetadata`] segment that the postscript locates by offset.

use flatbuffers::FlatBufferBuilder;
use flatbuffers::WIPOffset;
use flatbuffers::root;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_flatbuffers::FlatBufferRoot;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::footer as fb;
use vortex_utils::aliases::hash_map::HashMap;

/// User-defined file-level metadata, serialized into a single Vortex file segment.
pub(crate) struct FileMetadata {
    /// Entries sorted by key so the serialized bytes are deterministic.
    entries: Vec<(String, ByteBuffer)>,
}

impl FileMetadata {
    /// Build [`FileMetadata`] from a keyed map of opaque values.
    pub(crate) fn new(metadata: HashMap<String, ByteBuffer>) -> Self {
        let mut entries: Vec<(String, ByteBuffer)> = metadata.into_iter().collect();
        entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
        Self { entries }
    }

    /// Parse a metadata segment into a keyed map, copying values out so the file bytes aren't pinned.
    pub(crate) fn parse(bytes: &[u8]) -> VortexResult<HashMap<String, ByteBuffer>> {
        let fb = root::<fb::FileMetadata>(bytes)?;
        let mut map = HashMap::default();
        if let Some(entries) = fb.entries() {
            for entry in entries.iter() {
                let key = entry.key();
                if key.is_empty() {
                    vortex_bail!("File metadata contains an empty key");
                }
                let value = ByteBuffer::copy_from(entry.value().bytes());
                if map.insert(key.to_string(), value).is_some() {
                    vortex_bail!("File metadata contains duplicate key {key}");
                }
            }
        }
        Ok(map)
    }
}

impl FlatBufferRoot for FileMetadata {}

impl WriteFlatBuffer for FileMetadata {
    type Target<'a> = fb::FileMetadata<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
        let entries = self
            .entries
            .iter()
            .map(|(key, value)| {
                let key = fbb.create_string(key);
                let value = fbb.create_vector(value.as_slice());
                fb::MetadataEntry::create(
                    fbb,
                    &fb::MetadataEntryArgs {
                        key: Some(key),
                        value: Some(value),
                    },
                )
            })
            .collect::<Vec<_>>();
        let entries = fbb.create_vector(entries.as_slice());
        Ok(fb::FileMetadata::create(
            fbb,
            &fb::FileMetadataArgs {
                entries: Some(entries),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_flatbuffers::WriteFlatBufferExt;

    use super::*;

    fn metadata<const N: usize>(entries: [(&str, &[u8]); N]) -> FileMetadata {
        FileMetadata::new(
            entries
                .into_iter()
                .map(|(key, value)| (key.to_string(), ByteBuffer::copy_from(value)))
                .collect(),
        )
    }

    #[test]
    fn roundtrip_is_deterministic() -> VortexResult<()> {
        // Insertion order must not affect the serialized bytes (entries are sorted by key).
        let forward =
            metadata([("a", b"alpha"), ("b", b""), ("c", b"gamma")]).write_flatbuffer_bytes()?;
        let reverse =
            metadata([("c", b"gamma"), ("b", b""), ("a", b"alpha")]).write_flatbuffer_bytes()?;
        assert_eq!(forward.as_slice(), reverse.as_slice());

        let parsed = FileMetadata::parse(&forward)?;
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed["a"].as_slice(), b"alpha");
        assert!(parsed["b"].is_empty());
        Ok(())
    }

    #[test]
    fn parse_rejects_empty_key() -> VortexResult<()> {
        let bytes = metadata([("", b"value")]).write_flatbuffer_bytes()?;
        let error = FileMetadata::parse(&bytes).expect_err("empty key must be rejected");
        assert!(error.to_string().contains("empty key"));
        Ok(())
    }
}

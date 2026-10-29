// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex_array::flatbuffers::FlatBuffer;
use vortex_array::flatbuffers::FlatBufferRoot;
use vortex_array::flatbuffers::WriteFlatBuffer;
use vortex_array::flatbuffers::WriteFlatBufferExt;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_layout::LayoutContext;
use vortex_session::registry::ReadContext;
use vortex_utils::aliases::hash_map::HashMap;

use crate::EOF_SIZE;
use crate::Footer;
use crate::MAGIC_BYTES;
use crate::MAX_POSTSCRIPT_SIZE;
use crate::VERSION;
use crate::encryption::EncryptionSpec;
use crate::encryption::FileEncryption;
use crate::encryption::SegmentCipher;
use crate::encryption::SegmentRole;
use crate::footer::SegmentSpec;
use crate::footer::file_layout::FooterFlatBufferWriter;
use crate::footer::postscript::EncryptedStatistics;
use crate::footer::postscript::Postscript;
use crate::footer::postscript::PostscriptMetadata;
use crate::footer::postscript::PostscriptSegment;
use crate::footer::postscript::postscript_binding;

/// Serializes a [`Footer`] into footer buffers and the trailing postscript/EOF marker.
pub struct FooterSerializer {
    footer: Footer,
    metadata: HashMap<String, ByteBuffer>,
    exclude_dtype: bool,
    offset: u64,
    layout_ctx: LayoutContext,
    cipher: Option<SegmentCipher>,
    plaintext_footer: bool,
    /// Ciphers for the statistics of fields encrypted with the key at an index of the footer's
    /// encryption specs.
    key_ciphers: HashMap<u16, SegmentCipher>,
}

impl FooterSerializer {
    pub(super) fn new(footer: Footer) -> Self {
        Self {
            footer,
            metadata: HashMap::default(),
            exclude_dtype: false,
            offset: 0,
            layout_ctx: LayoutContext::default(),
            cipher: None,
            plaintext_footer: false,
            key_ciphers: HashMap::default(),
        }
    }

    /// Serialize layouts through `layout_ctx`.
    ///
    /// Every layout written to the file is interned here, so a context restricted with
    /// [`LayoutContext::with_allowed_ids`] fails the write rather than emitting a layout
    /// outside the permitted set. The default context permits any layout.
    pub fn with_layout_context(mut self, layout_ctx: LayoutContext) -> Self {
        self.layout_ctx = layout_ctx;
        self
    }

    /// Update the offset used to generate absolute segment locations.
    ///
    /// This represents the byte position that the first buffer emitted by this serializer will be
    /// written to.
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    /// Exclude the DType from the serialized footer.
    /// If excluded, the reader must be provided the DType from an external source.
    pub fn exclude_dtype(mut self) -> Self {
        self.exclude_dtype = true;
        self
    }

    /// Whether to exclude the DType from the serialized footer.
    /// If excluded, the reader must be provided the DType from an external source.
    pub fn with_exclude_dtype(mut self, exclude_dtype: bool) -> Self {
        self.exclude_dtype = exclude_dtype;
        self
    }

    /// Encrypt the dtype, layout, statistics, footer, and metadata segments with the footer key of
    /// `encryption`, or sign them if it configures a plaintext footer.
    ///
    /// This is required to serialize the footer of a file with encrypted segments, so that a
    /// standalone footer does not leak the file's metadata in plaintext.
    pub fn with_encryption(mut self, encryption: &FileEncryption) -> VortexResult<Self> {
        // The statistics of fields with their own key are encrypted with it. Match the footer's
        // specs to the configured keys by their key metadata.
        let keys = std::iter::once(encryption.footer_key()).chain(encryption.named_keys().values());
        let mut key_ciphers = HashMap::default();
        for (index, spec) in self.footer.encryption_specs().iter().enumerate().skip(1) {
            if let Some(key) = keys
                .clone()
                .find(|key| key.key_metadata() == spec.key_metadata())
            {
                key_ciphers.insert(
                    u16::try_from(index)?,
                    encryption.cipher_for_spec(key, spec)?,
                );
            }
        }
        // Readers require the footer segments to be protected by the footer's own footer key
        // spec, so reuse it when the footer describes an encrypted file.
        let footer_cipher = match key_ciphers.get(&1) {
            Some(cipher) => cipher.clone(),
            None if !self.footer.encryption_specs().is_empty() => vortex_bail!(
                "The footer key of the FileEncryption does not match the footer's footer key"
            ),
            None => {
                // Readers reject plaintext metadata segments in an encrypted file.
                if self.footer.metadata_segments().next().is_some() {
                    vortex_bail!(
                        "Cannot encrypt the footer of a file with plaintext metadata segments"
                    );
                }
                let cipher = encryption.file_cipher(encryption.footer_key())?;
                // Record the footer key, so that readers know that the file is encrypted. The
                // data segments keep index 0, as they are stored in plaintext.
                self.footer = self.footer.with_encryption_specs(
                    vec![EncryptionSpec::plaintext(), cipher.spec().clone()].into(),
                )?;
                key_ciphers.insert(1, cipher.clone());
                cipher
            }
        };
        Ok(self
            .with_cipher(footer_cipher, encryption.is_plaintext_footer())
            .with_key_ciphers(key_ciphers))
    }

    /// Encrypt the statistics of fields with their own key with the cipher at their key's index.
    pub(crate) fn with_key_ciphers(mut self, key_ciphers: HashMap<u16, SegmentCipher>) -> Self {
        self.key_ciphers = key_ciphers;
        self
    }

    /// Protect the footer segments with `cipher`, by signing them if `plaintext_footer`, and by
    /// encrypting them otherwise. User metadata segments are always encrypted.
    pub(crate) fn with_cipher(mut self, cipher: SegmentCipher, plaintext_footer: bool) -> Self {
        self.cipher = Some(cipher);
        self.plaintext_footer = plaintext_footer;
        self
    }

    pub(crate) fn with_metadata_segments(mut self, metadata: HashMap<String, ByteBuffer>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Serialize the footer into a byte buffer that can later be deserialized as a [`Footer`].
    /// This can be helpful for storing some footer data out-of-band to accelerate opening a file.
    pub fn serialize(self) -> VortexResult<Vec<ByteBuffer>> {
        self.serialize_with_metadata()
            .map(|(buffers, _metadata, _approx_byte_size)| buffers)
    }

    pub(crate) fn serialize_with_metadata(
        mut self,
    ) -> VortexResult<(Vec<ByteBuffer>, super::MetadataSegments, usize)> {
        let mut buffers = vec![];

        if !self.footer.encryption_specs().is_empty() && self.cipher.is_none() {
            vortex_bail!(
                "Footer describes encrypted segments; configure FooterSerializer::with_encryption \
                 to serialize it"
            );
        }
        let cipher = self.cipher.as_ref();
        let plaintext_footer = self.plaintext_footer;
        let protect = |role| match cipher {
            None => Protection::None,
            Some(cipher) if plaintext_footer => Protection::Sign(cipher, role),
            Some(cipher) => Protection::Encrypt(cipher, role),
        };
        let footer = &self.footer;

        let (metadata_segments, metadata_locators) = if self.metadata.is_empty() {
            let locators = footer
                .metadata_segments()
                .map(|(key, segment)| (key.to_string(), *segment))
                .collect::<Vec<_>>();
            let segments = locators
                .iter()
                .map(|(key, segment)| {
                    Ok(PostscriptMetadata {
                        key: key.clone(),
                        segment: PostscriptSegment {
                            offset: segment.offset,
                            length: segment.length,
                            alignment: segment.alignment,
                            encryption: footer.segment_encryption(segment)?,
                            signature: None,
                        },
                    })
                })
                .collect::<VortexResult<_>>()?;
            (segments, locators)
        } else {
            let encryption = match cipher {
                None => 0,
                Some(cipher) => footer
                    .encryption_specs()
                    .iter()
                    .position(|spec| spec == cipher.spec())
                    .and_then(|index| u16::try_from(index).ok())
                    .ok_or_else(|| {
                        vortex_err!("Footer encryption specs do not include the file's spec")
                    })?,
            };

            let mut metadata = std::mem::take(&mut self.metadata)
                .into_iter()
                .collect::<Vec<_>>();
            // Metadata is stored in a HashMap, so sort keys before assigning segment offsets.
            metadata.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));

            let mut segments = Vec::with_capacity(metadata.len());
            let mut locators = Vec::with_capacity(metadata.len());
            for (key, metadata) in metadata {
                let (metadata_buffers, segment) = write_buffer(
                    &mut self.offset,
                    metadata,
                    cipher.map(|cipher| (cipher, SegmentRole::Metadata(&key))),
                )?;
                buffers.extend(metadata_buffers);
                locators.push((
                    key.clone(),
                    SegmentSpec {
                        offset: segment.offset,
                        length: segment.length,
                        alignment: segment.alignment,
                        encryption,
                    },
                ));
                segments.push(PostscriptMetadata { key, segment });
            }
            (segments, locators)
        };
        let metadata_storage_bytes = buffers.iter().map(ByteBuffer::len).sum::<usize>();

        let dtype_segment = if self.exclude_dtype {
            None
        } else {
            let (buffer, dtype_segment) = write_flatbuffer(
                &mut self.offset,
                footer.dtype(),
                protect(SegmentRole::DType),
            )?;
            buffers.push(buffer);
            Some(dtype_segment)
        };

        // TODO(ngates): we should separate the read/write side of Context since the write side
        //  doesn't need to look anything up in the registry.
        let layout_ctx = self.layout_ctx.clone();

        let (buffer, layout_segment) = write_flatbuffer(
            &mut self.offset,
            &footer.layout().flatbuffer_writer(&layout_ctx),
            protect(SegmentRole::Layout),
        )?;
        buffers.push(buffer);

        let mut encrypted_statistics = Vec::new();
        let statistics_segment = match footer.statistics() {
            None => None,
            Some(stats) if stats.is_empty() => None,
            Some(stats) => {
                // The statistics of fields with their own key go into a segment per key, and are
                // left empty in the shared statistics segment.
                let encrypted_fields: Vec<u32> = footer
                    .statistics_encryption()
                    .iter()
                    .map(|(field, _)| *field)
                    .collect();
                // Entries are sorted by key, so their position identifies them to readers, which
                // read them before the footer that defines the key indices.
                for (position, (key, fields)) in footer
                    .statistics_encryption()
                    .iter()
                    .map(|(field, key)| (*key, *field))
                    .into_group_map()
                    .into_iter()
                    .sorted_unstable()
                    .enumerate()
                {
                    let cipher = self.key_ciphers.get(&key).ok_or_else(|| {
                        vortex_err!(
                            "Footer encrypts the statistics of fields {fields:?} with encryption \
                             spec {key}, whose key FooterSerializer::with_encryption was not given"
                        )
                    })?;
                    let (buffer, segment) = write_flatbuffer(
                        &mut self.offset,
                        &stats.retain_fields(&fields, footer.dtype()),
                        Protection::Encrypt(
                            cipher,
                            SegmentRole::KeyStatistics {
                                position: u16::try_from(position)?,
                                fields: &fields,
                            },
                        ),
                    )?;
                    buffers.push(buffer);
                    encrypted_statistics.push(EncryptedStatistics { fields, segment });
                }
                let shared = stats.with_field_stats(&encrypted_fields, None, footer.dtype());
                let (buffer, stats_segment) =
                    write_flatbuffer(&mut self.offset, &shared, protect(SegmentRole::Statistics))?;
                buffers.push(buffer);
                Some(stats_segment)
            }
        };

        // The footer authenticates the postscript entries it cannot otherwise protect.
        let binding = postscript_binding(
            dtype_segment.is_some(),
            statistics_segment.is_some(),
            metadata_segments
                .iter()
                .map(|entry: &PostscriptMetadata| entry.key.as_str()),
            encrypted_statistics
                .iter()
                .map(|entry: &EncryptedStatistics| entry.fields.as_slice()),
        );
        let (buffer, footer_segment) = write_flatbuffer(
            &mut self.offset,
            &FooterFlatBufferWriter {
                ctx: footer.array_read_ctx.clone(),
                layout_ctx: ReadContext::new(layout_ctx.to_ids()),
                segment_specs: Arc::clone(&footer.segments),
                encryption_specs: Arc::clone(footer.encryption_specs()),
            },
            protect(SegmentRole::Footer(&binding)),
        )?;
        buffers.push(buffer);

        // Assemble the postscript, and write it manually to avoid any framing.
        let postscript = Postscript {
            dtype: dtype_segment,
            layout: layout_segment,
            statistics: statistics_segment,
            footer: footer_segment,
            metadata: metadata_segments,
            encrypted_statistics,
        };
        let postscript_buffer = postscript.write_flatbuffer_bytes()?;
        if postscript_buffer.len() > MAX_POSTSCRIPT_SIZE as usize {
            Err(vortex_err!(
                "Postscript is too large ({} bytes); max postscript size is {}",
                postscript_buffer.len(),
                MAX_POSTSCRIPT_SIZE
            ))?;
        }

        let postscript_len = u16::try_from(postscript_buffer.len())
            .vortex_expect("Postscript already verified to fit into u16");
        buffers.push(postscript_buffer.into_inner());

        // And finally, the EOF 8-byte footer.
        let mut eof = [0u8; EOF_SIZE];
        eof[0..2].copy_from_slice(&VERSION.to_le_bytes());
        eof[2..4].copy_from_slice(&postscript_len.to_le_bytes());
        eof[4..8].copy_from_slice(&MAGIC_BYTES);
        buffers.push(ByteBuffer::copy_from(eof));

        let approx_byte_size =
            buffers.iter().map(ByteBuffer::len).sum::<usize>() - metadata_storage_bytes;
        Ok((buffers, metadata_locators.into(), approx_byte_size))
    }
}

/// How a footer segment is protected.
#[derive(Clone, Copy)]
enum Protection<'a> {
    None,
    Encrypt(&'a SegmentCipher, SegmentRole<'a>),
    Sign(&'a SegmentCipher, SegmentRole<'a>),
}

fn write_flatbuffer<F: FlatBufferRoot + WriteFlatBuffer>(
    offset: &mut u64,
    flatbuffer: &F,
    protection: Protection<'_>,
) -> VortexResult<(ByteBuffer, PostscriptSegment)> {
    let alignment = FlatBuffer::alignment();
    let mut buffer = flatbuffer.write_flatbuffer_bytes()?.into_inner();
    let mut spec = EncryptionSpec::plaintext();
    let mut signature = None;
    match protection {
        Protection::None => {}
        Protection::Encrypt(cipher, role) => {
            buffer = cipher.encrypt(role, &[buffer], alignment)?;
            spec = cipher.spec().clone();
        }
        Protection::Sign(cipher, role) => {
            signature = Some(Arc::from(cipher.sign(role, &buffer)?));
            spec = cipher.spec().clone();
        }
    }
    let length = u32::try_from(buffer.len())
        .map_err(|_| vortex_err!("flatbuffer length exceeds maximum u32"))?;

    let segment = PostscriptSegment {
        offset: *offset,
        length,
        alignment,
        encryption: spec,
        signature,
    };

    *offset += u64::from(length);

    Ok((buffer, segment))
}

fn write_buffer(
    offset: &mut u64,
    mut buffer: ByteBuffer,
    encryption: Option<(&SegmentCipher, SegmentRole<'_>)>,
) -> VortexResult<(Vec<ByteBuffer>, PostscriptSegment)> {
    let alignment = buffer.alignment();
    let mut spec = EncryptionSpec::plaintext();
    if let Some((cipher, role)) = encryption {
        buffer = cipher.encrypt(role, &[buffer], alignment)?;
        spec = cipher.spec().clone();
    }
    let length = u32::try_from(buffer.len())
        .map_err(|_| vortex_err!("metadata segment length exceeds maximum u32"))?;

    let padding = offset.next_multiple_of(alignment.as_usize() as u64) - *offset;
    let segment_offset = *offset + padding;

    let segment = PostscriptSegment {
        offset: segment_offset,
        length,
        alignment,
        encryption: spec,
        signature: None,
    };

    *offset += padding + u64::from(length);

    let mut buffers = Vec::with_capacity(if padding == 0 { 1 } else { 2 });
    if padding > 0 {
        buffers.push(ByteBuffer::zeroed(padding as usize));
    }
    buffers.push(buffer);

    Ok((buffers, segment))
}

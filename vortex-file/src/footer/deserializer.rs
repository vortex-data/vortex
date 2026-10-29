// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;
use std::sync::Arc;

use flatbuffers::root;
use vortex_array::dtype::DType;
use vortex_array::flatbuffers::FlatBuffer;
use vortex_array::flatbuffers::ReadFlatBuffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::EOF_SIZE;
use crate::Footer;
use crate::MAGIC_BYTES;
use crate::VERSION;
use crate::encryption::CipherResolver;
use crate::encryption::EncryptionSpec;
use crate::encryption::FileDecryption;
use crate::encryption::SegmentCipher;
use crate::encryption::SegmentRole;
use crate::footer::FileStatistics;
use crate::footer::SegmentSpec;
use crate::footer::postscript::EncryptedStatistics;
use crate::footer::postscript::Postscript;
use crate::footer::postscript::PostscriptSegment;

/// Deserialize a footer from the end of a Vortex file or created from a
/// [`crate::footer::FooterSerializer`].
///
/// The deserializer is incremental because callers may initially read only the tail of a file. Call
/// [`deserialize`](Self::deserialize) until it returns [`DeserializeStep::Done`]. If it asks for
/// [`DeserializeStep::NeedMoreData`], prefix the requested bytes with [`prefix_data`](Self::prefix_data).
/// If it asks for [`DeserializeStep::NeedFileSize`], call [`with_size`](Self::with_size) and retry.
/// If it asks for [`DeserializeStep::NeedKeys`], await [`resolve_keys`](Self::resolve_keys) and
/// retry.
pub struct FooterDeserializer {
    // A buffer representing the end of a Vortex file.
    // During deserialization, we may need to expand this buffer by requesting more data from
    // the caller.
    buffer: ByteBuffer,
    // The session to use for deserialization.
    session: VortexSession,
    // The DType, if provided externally.
    dtype: Option<DType>,
    // Resolves the ciphers that decrypt encrypted footer segments.
    resolver: Arc<CipherResolver>,

    // Internal state that we accumulate

    // The size of the file containing the serialized footer. For a standalone footer, this is the
    // size of the footer blob rather than the data file described by the footer.
    file_size: Option<u64>,
    // The postscript, once we've parsed it.
    postscript: Option<Postscript>,
}

impl FooterDeserializer {
    pub(super) fn new(initial_read: ByteBuffer, session: VortexSession) -> Self {
        Self {
            buffer: initial_read,
            session,
            dtype: None,
            resolver: CipherResolver::new(None),
            file_size: None,
            postscript: None,
        }
    }

    /// Decrypt the footer of an encrypted file.
    pub fn with_decryption(self, decryption: FileDecryption) -> Self {
        self.with_some_decryption(Some(decryption))
    }

    /// Provide or clear the [`FileDecryption`] of the footer.
    pub fn with_some_decryption(mut self, decryption: Option<FileDecryption>) -> Self {
        self.resolver = CipherResolver::new(decryption);
        self
    }

    /// Resolve keys through `resolver`, which the file's segment source shares, so that each key
    /// is retrieved once per opened file.
    pub(crate) fn with_resolver(mut self, resolver: Arc<CipherResolver>) -> Self {
        self.resolver = resolver;
        self
    }

    /// Retrieve the keys of the footer segments, when asked for with
    /// [`DeserializeStep::NeedKeys`].
    ///
    /// Failures are reported by the next call to [`deserialize`](Self::deserialize), and only for
    /// keys the footer cannot be read without.
    pub async fn resolve_keys(&mut self) {
        for (spec, _) in self.footer_specs() {
            // Errors are cached by the resolver, and surface when the footer is deserialized.
            drop(self.resolver.cipher(&spec).await);
        }
    }

    /// The encryption specs of the footer segments that must be decrypted, and whether the
    /// footer can be read without them.
    fn footer_specs(&self) -> Vec<(EncryptionSpec, bool)> {
        let Some(postscript) = &self.postscript else {
            return Vec::new();
        };
        let dtype = self
            .dtype
            .is_none()
            .then_some(&postscript.dtype)
            .into_iter()
            .flatten();
        let required = dtype
            .chain(&postscript.statistics)
            .chain([&postscript.layout, &postscript.footer])
            .map(|segment| (&segment.encryption, false));
        // Statistics encrypted with a field's key are left empty without the key, and readers
        // may choose not to retrieve field keys for them.
        let field_statistics = self
            .resolver
            .decryption()
            .is_some_and(|decryption| decryption.field_key_statistics);
        let optional = postscript
            .encrypted_statistics
            .iter()
            .filter(|_| field_statistics)
            .map(|entry| (&entry.segment.encryption, true));
        let mut specs: Vec<(EncryptionSpec, bool)> = Vec::new();
        for (spec, optional) in required.chain(optional) {
            if !spec.is_encrypted() {
                continue;
            }
            match specs.iter_mut().find(|(known, _)| known == spec) {
                Some((_, known_optional)) => *known_optional &= optional,
                None => specs.push((spec.clone(), optional)),
            }
        }
        specs
    }

    /// Provide the file dtype externally.
    ///
    /// This is required for files written with [`VortexWriteOptions::exclude_dtype`](crate::VortexWriteOptions::exclude_dtype).
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Provide or clear the externally known file dtype.
    pub fn with_some_dtype(mut self, dtype: Option<DType>) -> Self {
        self.dtype = dtype;
        self
    }

    /// Provide the size of the file containing this serialized footer.
    ///
    /// For a footer read from the end of a Vortex file, this is the file size. For a standalone
    /// blob created by [`crate::footer::FooterSerializer`] with its default offset, this is the
    /// size of that blob, not the size of the data file described by the footer.
    pub fn with_size(mut self, file_size: u64) -> Self {
        self.file_size = Some(file_size);
        self
    }

    /// Provide or clear the size of the file containing this serialized footer.
    pub fn with_some_size(mut self, file_size: Option<u64>) -> Self {
        self.file_size = file_size;
        self
    }

    /// Prefix more data to the existing buffer when requested by the deserializer.
    pub fn prefix_data(&mut self, more_data: ByteBuffer) {
        let mut buffer = ByteBufferMut::with_capacity(self.buffer.len() + more_data.len());
        buffer.extend_from_slice(&more_data);
        buffer.extend_from_slice(&self.buffer);
        self.buffer = buffer.freeze();
    }

    /// Advance footer deserialization.
    ///
    /// Returns the next missing input requirement or the finished [`Footer`].
    pub fn deserialize(&mut self) -> VortexResult<DeserializeStep> {
        let postscript = if let Some(postscript) = &self.postscript {
            postscript
        } else {
            self.postscript = Some(self.parse_postscript(&self.buffer)?);
            self.postscript
                .as_ref()
                .vortex_expect("Just set postscript")
        };

        // If we haven't been provided a DType, we must read one from the file.
        let dtype_segment = self
            .dtype
            .is_none()
            .then(|| {
                postscript.dtype.as_ref().ok_or_else(|| {
                    vortex_err!(
                        "Vortex file doesn't embed a DType and none provided to VortexOpenOptions"
                    )
                })
            })
            .transpose()?;

        // The other postscript segments are required, so now we figure out our the offset that
        // contains all the required segments.

        // The initial offset is the file size minus the size of our initial read.
        let Some(file_size) = self.file_size else {
            return Ok(DeserializeStep::NeedFileSize);
        };
        let initial_offset = file_size
            .checked_sub(self.buffer.len() as u64)
            .ok_or_else(|| {
                vortex_err!(
                    "Footer buffer length {} exceeds declared file size {file_size}",
                    self.buffer.len()
                )
            })?;

        let mut read_more_offset = initial_offset;
        if let Some(dtype_segment) = &dtype_segment {
            read_more_offset = read_more_offset.min(dtype_segment.offset);
        }
        if let Some(stats_segment) = &postscript.statistics {
            read_more_offset = read_more_offset.min(stats_segment.offset);
        }
        read_more_offset = read_more_offset.min(postscript.layout.offset);
        read_more_offset = read_more_offset.min(postscript.footer.offset);
        for entry in &postscript.encrypted_statistics {
            read_more_offset = read_more_offset.min(entry.segment.offset);
        }

        // Read more bytes if necessary.
        if read_more_offset < initial_offset {
            tracing::trace!(
                "Initial read from {initial_offset} did not cover all footer segments, reading from {read_more_offset}"
            );
            return Ok(DeserializeStep::NeedMoreData {
                offset: read_more_offset,
                len: usize::try_from(initial_offset - read_more_offset)?,
            });
        }

        // Retrieve the keys of the footer segments before reading them.
        if self
            .footer_specs()
            .iter()
            .any(|(spec, _)| self.resolver.resolved(spec).is_none())
        {
            return Ok(DeserializeStep::NeedKeys);
        }
        let postscript = self
            .postscript
            .as_ref()
            .vortex_expect("postscript parsed above");

        // Reject stripped protection before parsing any footer segment.
        self.validate_postscript_protection(postscript)?;

        // Now we read our initial segments.
        let segments = InitialSegments {
            initial_offset,
            initial_read: &self.buffer,
            resolver: &self.resolver,
        };
        let dtype = dtype_segment
            .map(|segment| {
                let bytes = segments.read(segment, SegmentRole::DType)?;
                DType::from_flatbuffer(bytes.into_flatbuffer(), &self.session)
            })
            .transpose()?
            .unwrap_or_else(|| self.dtype.clone().vortex_expect("DType was provided"));
        // File statistics are optional, so ones that can't be parsed are dropped rather than
        // failing the open. Ones that fail to decrypt were tampered with, which fails the open.
        let mut file_stats = match &postscript.statistics {
            None => None,
            Some(segment) => {
                let bytes = match segments.read(segment, SegmentRole::Statistics) {
                    Ok(bytes) => Some(bytes),
                    Err(error) if segment.encryption.is_encrypted() => return Err(error),
                    Err(error) => {
                        tracing::warn!(%error, "ignoring unreadable file statistics");
                        None
                    }
                };
                bytes.and_then(|bytes| {
                    root::<crate::flatbuffers::footer::FileStatistics>(&bytes)
                        .map_err(VortexError::from)
                        .and_then(|fb| FileStatistics::from_flatbuffer(&fb, &dtype, &self.session))
                        .inspect_err(
                            |error| tracing::warn!(%error, "ignoring unreadable file statistics"),
                        )
                        .ok()
                })
            }
        };
        let encrypted_statistics = self.read_encrypted_statistics(
            &segments,
            &postscript.encrypted_statistics,
            &dtype,
            &mut file_stats,
        )?;
        let metadata = postscript
            .metadata
            .iter()
            .map(|metadata| {
                let segment = SegmentSpec {
                    offset: metadata.segment.offset,
                    length: metadata.segment.length,
                    alignment: metadata.segment.alignment,
                    encryption: 0,
                };
                let end = segment
                    .offset
                    .checked_add(u64::from(segment.length))
                    .ok_or_else(|| {
                        vortex_err!("Metadata segment {} range overflowed u64", metadata.key)
                    })?;
                if end > file_size {
                    vortex_bail!(
                        "Metadata segment {} range {}..{} exceeds file size {}",
                        metadata.key,
                        segment.offset,
                        end,
                        file_size
                    );
                }
                let offset = usize::try_from(segment.offset)?;
                if !segment.alignment.is_offset_aligned(offset) {
                    vortex_bail!(
                        "Metadata segment {} offset {} is not aligned to {}",
                        metadata.key,
                        segment.offset,
                        segment.alignment
                    );
                }
                Ok((
                    metadata.key.clone(),
                    segment,
                    metadata.segment.encryption.clone(),
                ))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let binding = postscript.binding();
        let footer_bytes = segments.read(&postscript.footer, SegmentRole::Footer(&binding))?;
        let layout_bytes = segments.read(&postscript.layout, SegmentRole::Layout)?;
        let footer = Footer::from_flatbuffer(
            &footer_bytes,
            layout_bytes.into_flatbuffer(),
            dtype,
            file_stats,
            metadata,
            encrypted_statistics,
            &self.session,
        )?;
        self.validate_footer_protection(postscript, &footer)?;
        Ok(DeserializeStep::Done(footer))
    }

    /// Reject a footer whose segments are not protected by the footer key, although it describes
    /// an encrypted file.
    ///
    /// The postscript is not authenticated, so without this check, a reader configured for
    /// decryption would read footer segments that an attacker stripped of their encryption, and
    /// with them a forged layout, dtype, or statistics.
    fn validate_footer_protection(
        &self,
        postscript: &Postscript,
        footer: &Footer,
    ) -> VortexResult<()> {
        if self.resolver.decryption().is_none() || footer.encryption_specs().is_empty() {
            return Ok(());
        }
        let footer_key = footer.encryption_specs().get(1).ok_or_else(|| {
            vortex_err!("Encrypted file's footer lists no footer key encryption spec")
        })?;
        self.require_footer_protection(postscript, footer_key)
    }

    /// Reject an encrypted footer whose other footer segments are not protected like it, before
    /// any of them is parsed.
    ///
    /// [`validate_footer_protection`](Self::validate_footer_protection) then checks that this
    /// protection is the footer key's, once the footer is decrypted.
    fn validate_postscript_protection(&self, postscript: &Postscript) -> VortexResult<()> {
        if self.resolver.decryption().is_none() || !postscript.footer.encryption.is_encrypted() {
            return Ok(());
        }
        self.require_footer_protection(postscript, &postscript.footer.encryption)
    }

    /// Fail unless every footer segment the reader reads is protected with `spec`.
    fn require_footer_protection(
        &self,
        postscript: &Postscript,
        spec: &EncryptionSpec,
    ) -> VortexResult<()> {
        // An externally provided dtype is not read from the file.
        let dtype = self
            .dtype
            .is_none()
            .then_some(&postscript.dtype)
            .into_iter()
            .flatten();
        let segments = dtype
            .map(|segment| ("dtype", segment))
            .chain(
                postscript
                    .statistics
                    .iter()
                    .map(|segment| ("statistics", segment)),
            )
            .chain([
                ("layout", &postscript.layout),
                ("footer", &postscript.footer),
            ]);
        for (name, segment) in segments {
            if segment.encryption != *spec {
                vortex_bail!(
                    "The {name} segment of an encrypted Vortex file is not protected by its \
                     footer key; the file may have been tampered with"
                );
            }
        }
        Ok(())
    }

    /// Merge the statistics of fields encrypted with their own key into `file_stats`, where the
    /// key can be retrieved. Returns the fields and encryption spec of every entry.
    fn read_encrypted_statistics(
        &self,
        segments: &InitialSegments<'_>,
        entries: &[EncryptedStatistics],
        dtype: &DType,
        file_stats: &mut Option<FileStatistics>,
    ) -> VortexResult<Vec<(Vec<u32>, EncryptionSpec)>> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }
        let Some(fields) = dtype.as_struct_fields_opt() else {
            vortex_bail!("Encrypted statistics require a struct dtype, got {dtype}");
        };
        let mut seen = vec![false; fields.nfields()];
        let mut result = Vec::with_capacity(entries.len());
        for (position, entry) in entries.iter().enumerate() {
            let position = u16::try_from(position)?;
            for &field in &entry.fields {
                let seen = seen.get_mut(field as usize).ok_or_else(|| {
                    vortex_err!("Encrypted statistics refer to missing field {field}")
                })?;
                if std::mem::replace(seen, true) {
                    vortex_bail!("Statistics of field {field} are encrypted more than once");
                }
            }

            // The key of a field's statistics may be unavailable to a reader that cannot read
            // the field either, so leave its statistics empty rather than failing.
            match segments.cipher(&entry.segment.encryption) {
                Ok(cipher) => {
                    let bytes = segments.read_with(
                        &entry.segment,
                        SegmentRole::KeyStatistics {
                            position,
                            fields: &entry.fields,
                        },
                        &cipher,
                    )?;
                    // Like the shared statistics, ones that can't be parsed are dropped.
                    match root::<crate::flatbuffers::footer::FileStatistics>(&bytes)
                        .map_err(VortexError::from)
                        .and_then(|fb| FileStatistics::from_flatbuffer(&fb, dtype, &self.session))
                    {
                        Ok(stats) => {
                            let merged = file_stats
                                .take()
                                .unwrap_or_else(|| FileStatistics::empty_with_dtype(dtype));
                            *file_stats =
                                Some(merged.with_field_stats(&entry.fields, Some(&stats), dtype));
                        }
                        Err(error) => {
                            tracing::warn!(%error, "ignoring unreadable encrypted file statistics");
                        }
                    }
                }
                Err(err) => {
                    tracing::debug!("Statistics of fields {:?} unavailable: {err}", entry.fields);
                }
            }
            result.push((entry.fields.clone(), entry.segment.encryption.clone()));
        }
        Ok(result)
    }

    /// The current buffer being used for deserialization.
    pub fn buffer(&self) -> &ByteBuffer {
        &self.buffer
    }

    /// Parse the postscript from the initial read.
    fn parse_postscript(&self, initial_read: &[u8]) -> VortexResult<Postscript> {
        if initial_read.len() < EOF_SIZE {
            vortex_bail!(
                "Initial read must be at least EOF_SIZE ({}) bytes",
                EOF_SIZE
            );
        }
        let eof_loc = initial_read.len() - EOF_SIZE;
        let magic_bytes_loc = eof_loc + (EOF_SIZE - MAGIC_BYTES.len());

        let magic_number = &initial_read[magic_bytes_loc..];
        if magic_number != MAGIC_BYTES {
            vortex_bail!("Malformed file, invalid magic bytes, got {magic_number:?}")
        }

        let version = u16::from_le_bytes(
            initial_read[eof_loc..eof_loc + 2]
                .try_into()
                .map_err(|e| vortex_err!("Version was not a u16 {e}"))?,
        );
        if version != VERSION {
            vortex_bail!("Malformed file, unsupported version {version}")
        }

        let ps_size = u16::from_le_bytes(
            initial_read[eof_loc + 2..eof_loc + 4]
                .try_into()
                .map_err(|e| vortex_err!("Postscript size was not a u16 {e}"))?,
        ) as usize;

        if initial_read.len() < ps_size + EOF_SIZE {
            vortex_bail!(
                "Initial read must be at least {} bytes to include the Postscript",
                ps_size + EOF_SIZE
            );
        }

        Postscript::read_flatbuffer_bytes(&initial_read[eof_loc - ps_size..eof_loc])
    }
}

/// The footer segments covered by the deserializer's buffer.
struct InitialSegments<'a> {
    initial_offset: u64,
    initial_read: &'a [u8],
    resolver: &'a CipherResolver,
}

impl<'a> InitialSegments<'a> {
    /// The resolved cipher of an encrypted spec.
    fn cipher(&self, spec: &EncryptionSpec) -> VortexResult<Arc<SegmentCipher>> {
        self.resolver
            .resolved(spec)
            .ok_or_else(|| {
                vortex_err!("Footer key not resolved; await FooterDeserializer::resolve_keys")
            })??
            .ok_or_else(|| vortex_err!("Encrypted footer segment has a plaintext encryption spec"))
    }

    /// Slice a segment out of the buffer, decrypting it with `cipher`.
    fn read_with(
        &self,
        segment: &PostscriptSegment,
        role: SegmentRole<'_>,
        cipher: &SegmentCipher,
    ) -> VortexResult<ByteBuffer> {
        let bytes = checked_segment_slice(self.initial_read, self.initial_offset, segment)?;
        cipher.decrypt(role, ByteBuffer::copy_from(bytes), segment.alignment)
    }

    /// Slice a segment out of the buffer, decrypting it if necessary.
    fn read(
        &self,
        segment: &PostscriptSegment,
        role: SegmentRole<'_>,
    ) -> VortexResult<SegmentBytes<'a>> {
        let bytes = checked_segment_slice(self.initial_read, self.initial_offset, segment)?;
        if !segment.encryption.is_encrypted() {
            return Ok(SegmentBytes::Borrowed(bytes));
        }
        let cipher = self.cipher(&segment.encryption)?;
        let plaintext = cipher.decrypt(role, ByteBuffer::copy_from(bytes), segment.alignment)?;
        Ok(SegmentBytes::Decrypted(plaintext))
    }
}

/// A footer segment, borrowed from the read buffer, or decrypted into a buffer of its own.
enum SegmentBytes<'a> {
    Borrowed(&'a [u8]),
    Decrypted(ByteBuffer),
}

impl SegmentBytes<'_> {
    /// The segment as a flatbuffer, copied only if it is borrowed or misaligned.
    fn into_flatbuffer(self) -> FlatBuffer {
        match self {
            SegmentBytes::Borrowed(bytes) => FlatBuffer::copy_from(bytes),
            SegmentBytes::Decrypted(buffer) => FlatBuffer::align_from(buffer),
        }
    }
}

impl Deref for SegmentBytes<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            SegmentBytes::Borrowed(bytes) => bytes,
            SegmentBytes::Decrypted(buffer) => buffer.as_slice(),
        }
    }
}

fn checked_segment_slice<'a>(
    read: &'a [u8],
    read_offset: u64,
    segment: &PostscriptSegment,
) -> VortexResult<&'a [u8]> {
    let offset = usize::try_from(segment.offset.checked_sub(read_offset).ok_or_else(|| {
        vortex_err!(
            "Segment offset {} is smaller than file read offset {read_offset}",
            segment.offset
        )
    })?)?;
    offset
        .checked_add(segment.length as usize)
        .and_then(|end| read.get(offset..end))
        .ok_or_else(|| {
            vortex_err!(
                "Segment length {} (at offset {}) out of bounds of slice of length {}",
                segment.length,
                offset,
                read.len()
            )
        })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::flatbuffers::WriteFlatBufferExt;
    use vortex_session::VortexSession;

    use super::*;

    fn segment(offset: u64, length: u32) -> PostscriptSegment {
        PostscriptSegment {
            offset,
            length,
            alignment: FlatBuffer::alignment(),
            encryption: EncryptionSpec::plaintext(),
        }
    }

    #[test]
    fn in_bounds_segment_slice() -> VortexResult<()> {
        let read: Vec<u8> = (0u8..10).collect();
        let sliced = checked_segment_slice(&read, 100, &segment(104, 4))?;
        assert_eq!(sliced, &read[4..8]);
        Ok(())
    }

    #[rstest]
    #[case::offset_before_read_start(100, 99, 4, "smaller than file read offset")]
    #[case::end_past_buffer(100, 105, 6, "out of bounds")]
    #[case::offset_past_buffer(100, 120, 1, "out of bounds")]
    #[case::end_overflows_usize(0, u64::MAX, u32::MAX, "out of bounds")]
    fn out_of_bounds_segment_slice(
        #[case] read_offset: u64,
        #[case] segment_offset: u64,
        #[case] segment_length: u32,
        #[case] expected: &str,
    ) {
        let read = [0u8; 10];
        let err =
            checked_segment_slice(&read, read_offset, &segment(segment_offset, segment_length))
                .unwrap_err();
        assert!(err.to_string().contains(expected), "{err}");
    }

    fn eof_buffer(postscript: &Postscript) -> VortexResult<ByteBuffer> {
        let postscript_bytes = postscript.write_flatbuffer_bytes()?;
        let mut buffer = ByteBufferMut::with_capacity(postscript_bytes.len() + EOF_SIZE);
        buffer.extend_from_slice(&postscript_bytes);
        buffer.extend_from_slice(&VERSION.to_le_bytes());
        buffer.extend_from_slice(&u16::try_from(postscript_bytes.len())?.to_le_bytes());
        buffer.extend_from_slice(&MAGIC_BYTES);
        Ok(buffer.freeze())
    }

    #[rstest]
    #[case::length_past_eof(segment(0, u32::MAX))]
    #[case::offset_overflow(segment(u64::MAX, u32::MAX))]
    fn deserialize_rejects_out_of_bounds_footer_segment(
        #[case] footer_segment: PostscriptSegment,
    ) -> VortexResult<()> {
        let postscript = Postscript {
            dtype: None,
            layout: segment(0, 1),
            statistics: None,
            footer: footer_segment,
            metadata: Vec::new(),
            encrypted_statistics: Vec::new(),
        };
        let buffer = eof_buffer(&postscript)?;
        let file_size = buffer.len() as u64;

        let mut deserializer = FooterDeserializer::new(buffer, VortexSession::empty())
            .with_dtype(DType::Primitive(PType::I32, Nullability::NonNullable))
            .with_size(file_size);
        let err = deserializer.deserialize().unwrap_err();
        assert!(err.to_string().contains("out of bounds"), "{err}");
        Ok(())
    }

    #[test]
    fn deserialize_rejects_buffer_larger_than_declared_size() -> VortexResult<()> {
        let postscript = Postscript {
            dtype: None,
            layout: segment(0, 1),
            statistics: None,
            footer: segment(1, 1),
            metadata: Vec::new(),
            encrypted_statistics: Vec::new(),
        };
        let buffer = eof_buffer(&postscript)?;
        let declared_size = buffer.len() as u64 - 1;

        let mut deserializer = FooterDeserializer::new(buffer, VortexSession::empty())
            .with_dtype(DType::Primitive(PType::I32, Nullability::NonNullable))
            .with_size(declared_size);
        let err = deserializer.deserialize().unwrap_err();
        assert!(err.to_string().contains("exceeds declared"), "{err}");
        Ok(())
    }
}

#[derive(Debug)]
/// Result of one [`FooterDeserializer::deserialize`] step.
pub enum DeserializeStep {
    /// Additional data needed to continue deserialization.
    NeedMoreData {
        /// Absolute file offset to read from.
        offset: u64,
        /// Number of bytes to read and prefix into the deserializer.
        len: usize,
    },
    /// The total file size is required before offsets can be resolved.
    NeedFileSize,
    /// The keys of encrypted footer segments must be retrieved, with
    /// [`FooterDeserializer::resolve_keys`].
    NeedKeys,
    /// Footer deserialization is complete.
    Done(Footer),
}

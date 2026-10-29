// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;

use crate::Footer;
use crate::SegmentSpec;
use crate::encryption::CipherResolver;
use crate::encryption::EncryptionSpec;
use crate::encryption::SegmentRole;

/// A [`SegmentSource`] that decrypts the encrypted segments of a file.
///
/// It sits above any segment cache, so that caches only hold ciphertext, unless the reader asked
/// for a plaintext cache. Keys are retrieved when a segment that needs them is first requested.
pub(crate) struct DecryptingSegmentSource {
    inner: Arc<dyn SegmentSource>,
    /// The segment map followed by the metadata segments, indexed by segment id.
    segments: Arc<[SegmentSpec]>,
    /// The footer's encryption specs, indexed by [`SegmentSpec::encryption`].
    specs: Arc<[EncryptionSpec]>,
    resolver: Arc<CipherResolver>,
    /// The number of segments in the segment map. Later ids are metadata segments.
    num_data_segments: usize,
    /// The keys of the metadata segments, in segment id order.
    metadata_keys: Arc<[String]>,
}

impl DecryptingSegmentSource {
    /// Wrap `inner` to decrypt the segments of `footer`, or return it unchanged if the file is not
    /// encrypted.
    pub(crate) fn wrap(
        inner: Arc<dyn SegmentSource>,
        footer: &Footer,
        resolver: &Arc<CipherResolver>,
    ) -> VortexResult<Arc<dyn SegmentSource>> {
        Ok(match Self::new(Arc::clone(&inner), footer, resolver)? {
            Some(decrypting) => Arc::new(decrypting),
            None => inner,
        })
    }

    /// Create a source decrypting the segments of `footer` read from `inner`, or `None` if the
    /// file is not encrypted.
    pub(crate) fn new(
        inner: Arc<dyn SegmentSource>,
        footer: &Footer,
        resolver: &Arc<CipherResolver>,
    ) -> VortexResult<Option<Self>> {
        // Writers only record encryption specs for files written with encryption.
        if footer.encryption_specs().is_empty() {
            if resolver
                .decryption()
                .is_some_and(|decryption| !decryption.allow_plaintext_files)
            {
                vortex_bail!(
                    "Vortex file is not encrypted, but decryption was configured; allow plaintext \
                     files with FileDecryption::with_plaintext_files_allowed"
                );
            }
            return Ok(None);
        }
        Ok(Some(Self {
            inner,
            segments: footer.segment_specs_with_metadata(),
            specs: Arc::clone(footer.encryption_specs()),
            resolver: Arc::clone(resolver),
            num_data_segments: footer.segment_map().len(),
            metadata_keys: footer
                .metadata_segments()
                .map(|(key, _)| key.to_string())
                .collect(),
        }))
    }

    /// The segment spec and encryption spec of an encrypted segment, or `None` for a plaintext
    /// or unknown segment.
    fn encryption(&self, id: SegmentId) -> Option<(&SegmentSpec, &EncryptionSpec)> {
        let segment = self.segments.get(*id as usize)?;
        let spec = self.specs.get(usize::from(segment.encryption))?;
        spec.is_encrypted().then_some((segment, spec))
    }

    /// The key of the metadata segment `id`, or `None` for a data segment.
    fn metadata_key(&self, id: SegmentId) -> Option<&str> {
        (*id as usize)
            .checked_sub(self.num_data_segments)
            .and_then(|metadata| self.metadata_keys.get(metadata))
            .map(String::as_str)
    }

    /// Decrypt segment `id`, read as `encrypted`.
    async fn decrypt(
        resolver: &CipherResolver,
        spec: &EncryptionSpec,
        id: SegmentId,
        metadata_key: Option<&str>,
        alignment: Alignment,
        encrypted: ByteBuffer,
    ) -> VortexResult<ByteBuffer> {
        let cipher = resolver
            .cipher(spec)
            .await
            .map_err(|err| vortex_err!("Cannot decrypt segment {}: {err}", *id))?
            .ok_or_else(|| vortex_err!("Encrypted segment {} has a plaintext spec", *id))?;
        let role = match metadata_key {
            Some(key) => SegmentRole::Metadata(key),
            None => SegmentRole::Data(*id),
        };
        cipher.decrypt(role, encrypted, alignment)
    }

    /// Decrypt a segment covered by the footer's initial read, for a cache that holds plaintext.
    ///
    /// Returns `None` if the segment cannot be decrypted, so that it is left out of the cache and
    /// the error surfaces when the segment is requested.
    pub(crate) async fn decrypt_initial(
        &self,
        id: SegmentId,
        buffer: ByteBuffer,
    ) -> Option<ByteBuffer> {
        let Some((segment, spec)) = self.encryption(id) else {
            return Some(buffer);
        };
        let decrypted = Self::decrypt(
            &self.resolver,
            spec,
            id,
            self.metadata_key(id),
            segment.alignment,
            buffer,
        )
        .await;
        decrypted.ok()
    }
}

impl SegmentSource for DecryptingSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let Some((segment, spec)) = self.encryption(id) else {
            // Plaintext segments, and unknown ids for the inner source to reject.
            return self.inner.request(id);
        };
        let (alignment, spec) = (segment.alignment, spec.clone());
        let metadata_key = self.metadata_key(id).map(str::to_string);
        let resolver = Arc::clone(&self.resolver);
        // Request the segment now, so that the read is registered for coalescing while the key
        // is retrieved.
        let read = self.inner.request(id);
        async move {
            let encrypted = read.await?.try_into_host()?.await?;
            let plaintext = Self::decrypt(
                &resolver,
                &spec,
                id,
                metadata_key.as_deref(),
                alignment,
                encrypted,
            )
            .await?;
            Ok(BufferHandle::new_host(plaintext))
        }
        .boxed()
    }
}

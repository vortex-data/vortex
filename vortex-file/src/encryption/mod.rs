// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Segment-level encryption of Vortex files.
//!
//! Encryption operates on whole segments: data segments, and the dtype, layout, statistics,
//! footer, and user metadata segments referenced by the postscript. The postscript itself stays in
//! plaintext, since it only records segment locations and the [`EncryptionSpec`] of each.
//!
//! Writers configure a [`FileEncryption`] with a footer key. Keys are never written to the file.
//! Each key's opaque key metadata is stored instead, so that readers configured with a
//! [`FileDecryption`] can resolve the key again through a [`KeyRetriever`].
//!
//! Every segment, including the footer, is encrypted with the footer key, which also hides the
//! schema.
//!
//! Segments are encrypted with AES-GCM. The additional authenticated data binds every segment to its
//! file, through random per-file bytes, and to its position in the file, so that segments cannot
//! be swapped without failing decryption. An optional AAD prefix additionally binds the file to an
//! application-defined identity, such as its path, so that a file cannot be replaced by another
//! file encrypted with the same key. The prefix is not stored in the file, so readers must supply
//! it.

mod cipher;
mod resolver;

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use async_trait::async_trait;
use flatbuffers::FlatBufferBuilder;
use flatbuffers::WIPOffset;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use zeroize::Zeroize;

pub(crate) use self::cipher::*;
pub(crate) use self::resolver::*;
use crate::flatbuffers::footer as fb;

/// The supported AES key lengths in bytes: AES-128, AES-192, and AES-256.
pub const KEY_LENGTHS: [usize; 3] = [16, 24, 32];
const AAD_FILE_UNIQUE_LEN: usize = 16;

/// An AES-128, AES-192, or AES-256 key used to encrypt and decrypt Vortex file segments.
///
/// The key bytes are zeroed on drop and are not printed by [`Debug`].
#[derive(Clone)]
pub struct EncryptionKey(Box<[u8]>);

impl EncryptionKey {
    /// Create a key from a slice of 16, 24, or 32 bytes.
    pub fn try_from_slice(key: &[u8]) -> VortexResult<Self> {
        if !KEY_LENGTHS.contains(&key.len()) {
            vortex_bail!(
                "Encryption key must be 16, 24, or 32 bytes, but got {} bytes",
                key.len()
            );
        }
        Ok(Self(key.into()))
    }

    /// Generate a random key of `len` bytes, which must be 16, 24, or 32.
    pub fn generate(len: usize) -> VortexResult<Self> {
        let mut key = vec![0u8; len];
        fill_random(&mut key)?;
        let result = Self::try_from_slice(&key);
        key.zeroize();
        result
    }

    /// The raw key bytes, e.g. to wrap the key with a key management service.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl From<[u8; 16]> for EncryptionKey {
    fn from(key: [u8; 16]) -> Self {
        Self(key.into())
    }
}

impl From<[u8; 24]> for EncryptionKey {
    fn from(key: [u8; 24]) -> Self {
        Self(key.into())
    }
}

impl From<[u8; 32]> for EncryptionKey {
    fn from(key: [u8; 32]) -> Self {
        Self(key.into())
    }
}

impl Drop for EncryptionKey {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl Debug for EncryptionKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "EncryptionKey(<{} redacted bytes>)", self.0.len())
    }
}

/// Resolves the [`EncryptionKey`] for the key metadata stored in an encrypted file.
///
/// Implementations typically look the key up in a key management service. Each key is retrieved
/// at most once per opened file: the keys of the footer and of encrypted file statistics when the
/// footer is read, and other keys when a segment encrypted with them is first read.
/// Implementations must return an error if the key is unknown. A wrong key is detected when the
/// first segment fails to decrypt. An error is kept for the rest of the opened file, so every
/// segment that needs the key fails with it, and a transient failure, such as an unavailable key
/// management service, requires the file to be opened again.
///
/// The key metadata comes from the file, which may have been tampered with: only return keys
/// that the file is allowed to be encrypted with, since a key that an attacker also holds lets
/// them forge the file's content.
#[async_trait]
pub trait KeyRetriever: Send + Sync {
    /// Return the key identified by `key_metadata`.
    async fn retrieve_key(&self, key_metadata: &[u8]) -> VortexResult<EncryptionKey>;
}

/// A [`KeyRetriever`] from a function that resolves keys without blocking.
#[async_trait]
impl<F> KeyRetriever for F
where
    F: Fn(&[u8]) -> VortexResult<EncryptionKey> + Send + Sync,
{
    async fn retrieve_key(&self, key_metadata: &[u8]) -> VortexResult<EncryptionKey> {
        self(key_metadata)
    }
}

/// A [`KeyRetriever`] that returns the same key for any key metadata.
#[async_trait]
impl KeyRetriever for EncryptionKey {
    async fn retrieve_key(&self, _key_metadata: &[u8]) -> VortexResult<EncryptionKey> {
        Ok(self.clone())
    }
}

/// A key and the metadata that identifies it to readers.
#[derive(Clone, Debug)]
pub struct KeyConfig {
    key: EncryptionKey,
    key_metadata: Arc<[u8]>,
}

impl KeyConfig {
    /// Configure `key`, identified by `key_metadata`.
    ///
    /// `key_metadata` is stored in plaintext in the file and passed to the reader's
    /// [`KeyRetriever`] to resolve the key again, e.g. a key id in a key management service. It
    /// must not contain the key itself.
    ///
    /// Every segment is encrypted with a random 96-bit nonce, so a key must not encrypt more than
    /// about 2<sup>32</sup> segments across all files. Prefer a fresh key for every file, such as
    /// a data key generated for the file and wrapped by a key management service, over one key
    /// shared by many files.
    pub fn new(key: EncryptionKey, key_metadata: impl Into<Arc<[u8]>>) -> Self {
        Self {
            key,
            key_metadata: key_metadata.into(),
        }
    }

    /// The key.
    pub(crate) fn key(&self) -> &EncryptionKey {
        &self.key
    }

    /// The metadata that identifies the key to readers.
    pub fn key_metadata(&self) -> &[u8] {
        &self.key_metadata
    }

    /// Create the cipher for one file, with fresh random per-file AAD bytes.
    pub(crate) fn file_cipher(
        &self,
        algorithm: EncryptionAlgorithm,
        aad_prefix: Option<&[u8]>,
    ) -> VortexResult<SegmentCipher> {
        if algorithm == EncryptionAlgorithm::None {
            vortex_bail!("EncryptionAlgorithm::None cannot be used to encrypt a file");
        }
        let mut aad_file_unique = [0u8; AAD_FILE_UNIQUE_LEN];
        fill_random(&mut aad_file_unique)?;
        let spec = EncryptionSpec {
            algorithm,
            key_metadata: Arc::clone(&self.key_metadata),
            aad_file_unique: Arc::from(aad_file_unique),
            requires_aad_prefix: aad_prefix.is_some(),
        };
        SegmentCipher::new(spec, self.key.clone(), aad_prefix.unwrap_or_default())
    }
}

/// Configures how a Vortex file is encrypted when it is written.
///
/// By default, every segment of the file, including the footer, is encrypted with AES-GCM under
/// the footer key. Readers configure the matching [`FileDecryption`].
///
/// Writing an encrypted file needs a source of random bytes. On `wasm32-unknown-unknown`, the
/// `getrandom` backend must be configured for this, otherwise the write fails. Reading encrypted
/// files needs no randomness.
#[derive(Clone, Debug)]
pub struct FileEncryption {
    footer_key: KeyConfig,
    algorithm: EncryptionAlgorithm,
    aad_prefix: Option<Arc<[u8]>>,
}

impl FileEncryption {
    /// Encrypt every segment of the file, including the footer, with `footer_key`.
    pub fn new(footer_key: KeyConfig) -> Self {
        Self {
            footer_key,
            algorithm: EncryptionAlgorithm::AesGcm,
            aad_prefix: None,
        }
    }

    /// Set the encryption algorithm. Defaults to, and currently only supports,
    /// [`EncryptionAlgorithm::AesGcm`].
    pub fn with_algorithm(mut self, algorithm: EncryptionAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Bind the encrypted segments of the file to an application-defined AAD prefix, such as the
    /// file's path.
    ///
    /// The prefix is not stored in the file, so readers must supply it with
    /// [`FileDecryption::with_aad_prefix`]. It applies to every key of the file.
    pub fn with_aad_prefix(mut self, aad_prefix: impl Into<Arc<[u8]>>) -> Self {
        self.aad_prefix = Some(aad_prefix.into());
        self
    }

    /// The footer key.
    pub(crate) fn footer_key(&self) -> &KeyConfig {
        &self.footer_key
    }

    /// Create the cipher of `key` for one file.
    pub(crate) fn file_cipher(&self, key: &KeyConfig) -> VortexResult<SegmentCipher> {
        key.file_cipher(self.algorithm, self.aad_prefix.as_deref())
    }

    /// Create the cipher of `key` for the segments of an existing file that use `spec`.
    pub(crate) fn cipher_for_spec(
        &self,
        key: &KeyConfig,
        spec: &EncryptionSpec,
    ) -> VortexResult<SegmentCipher> {
        if spec.requires_aad_prefix != self.aad_prefix.is_some() {
            vortex_bail!(
                "The file's encryption spec and the FileEncryption disagree on whether an AAD \
                 prefix is used"
            );
        }
        SegmentCipher::new(
            spec.clone(),
            key.key().clone(),
            self.aad_prefix.as_deref().unwrap_or_default(),
        )
    }
}

/// Configures how an encrypted Vortex file is decrypted when it is read.
#[derive(Clone)]
pub struct FileDecryption {
    pub(crate) keys: Arc<dyn KeyRetriever>,
    pub(crate) aad_prefix: Option<Arc<[u8]>>,
    pub(crate) allow_plaintext_files: bool,
    pub(crate) cache_plaintext: bool,
}

impl FileDecryption {
    /// Decrypt files with the keys resolved by `keys`.
    pub fn new(keys: Arc<dyn KeyRetriever>) -> Self {
        Self {
            keys,
            aad_prefix: None,
            allow_plaintext_files: false,
            cache_plaintext: false,
        }
    }

    /// Supply the AAD prefix the file was bound to, such as its path.
    ///
    /// This is required to read files written with [`FileEncryption::with_aad_prefix`], and
    /// reading fails for files without a prefix. A wrong prefix fails decryption.
    pub fn with_aad_prefix(mut self, aad_prefix: impl Into<Arc<[u8]>>) -> Self {
        self.aad_prefix = Some(aad_prefix.into());
        self
    }

    /// Decrypt segments before they enter the segment cache, rather than after they leave it.
    ///
    /// By default the cache holds ciphertext, so every scan decrypts the segments it reads again.
    /// With this option, repeated scans of a file find plaintext in the cache. The cache then
    /// holds the file's plaintext, so only use it with a cache that is private to readers allowed
    /// to see every key's data, such as one configured with
    /// [`VortexOpenOptions::with_segment_cache`](crate::VortexOpenOptions::with_segment_cache)
    /// for a single file.
    pub fn with_plaintext_cache(mut self) -> Self {
        self.cache_plaintext = true;
        self
    }

    /// Allow reading files that are not encrypted.
    ///
    /// By default, a reader configured for decryption rejects unencrypted files, so that an
    /// attacker cannot replace an encrypted file with a plaintext one.
    pub fn with_plaintext_files_allowed(mut self) -> Self {
        self.allow_plaintext_files = true;
        self
    }
}

impl Debug for FileDecryption {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileDecryption")
            .field("aad_prefix", &self.aad_prefix)
            .field("allow_plaintext_files", &self.allow_plaintext_files)
            .field("cache_plaintext", &self.cache_plaintext)
            .finish_non_exhaustive()
    }
}

/// The encryption algorithm of a segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EncryptionAlgorithm {
    /// The segment is stored in plaintext.
    None,
    /// Every segment is encrypted with AES-GCM, which detects modified segments.
    AesGcm,
}

/// Describes how a segment is encrypted, without the key itself.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EncryptionSpec {
    algorithm: EncryptionAlgorithm,
    key_metadata: Arc<[u8]>,
    aad_file_unique: Arc<[u8]>,
    requires_aad_prefix: bool,
}

impl EncryptionSpec {
    /// The spec of a plaintext segment.
    pub fn plaintext() -> Self {
        Self {
            algorithm: EncryptionAlgorithm::None,
            key_metadata: Arc::from([]),
            aad_file_unique: Arc::from([]),
            requires_aad_prefix: false,
        }
    }

    /// The encryption algorithm.
    pub fn algorithm(&self) -> EncryptionAlgorithm {
        self.algorithm
    }

    /// The opaque metadata used to resolve the key with a [`KeyRetriever`].
    pub fn key_metadata(&self) -> &[u8] {
        &self.key_metadata
    }

    /// Whether readers must supply the AAD prefix the segments were encrypted with.
    pub fn requires_aad_prefix(&self) -> bool {
        self.requires_aad_prefix
    }

    /// Whether segments with this spec are encrypted.
    pub fn is_encrypted(&self) -> bool {
        self.algorithm != EncryptionAlgorithm::None
    }

    pub(crate) fn read_flatbuffer(fb: &fb::EncryptionSpec<'_>) -> VortexResult<Self> {
        let algorithm = match fb.algorithm() {
            fb::EncryptionAlgorithm::None => EncryptionAlgorithm::None,
            fb::EncryptionAlgorithm::AesGcm => EncryptionAlgorithm::AesGcm,
            other => vortex_bail!("Unsupported encryption algorithm {}", other.0),
        };
        Ok(Self {
            algorithm,
            key_metadata: fb
                .key_metadata()
                .map(|bytes| Arc::from(bytes.bytes()))
                .unwrap_or_else(|| Arc::from([])),
            aad_file_unique: fb
                .aad_file_unique()
                .map(|bytes| Arc::from(bytes.bytes()))
                .unwrap_or_else(|| Arc::from([])),
            requires_aad_prefix: fb.requires_aad_prefix(),
        })
    }

    pub(crate) fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<fb::EncryptionSpec<'fb>> {
        let algorithm = match self.algorithm {
            EncryptionAlgorithm::None => fb::EncryptionAlgorithm::None,
            EncryptionAlgorithm::AesGcm => fb::EncryptionAlgorithm::AesGcm,
        };
        let key_metadata =
            (!self.key_metadata.is_empty()).then(|| fbb.create_vector(&self.key_metadata));
        let aad_file_unique =
            (!self.aad_file_unique.is_empty()).then(|| fbb.create_vector(&self.aad_file_unique));
        fb::EncryptionSpec::create(
            fbb,
            &fb::EncryptionSpecArgs {
                algorithm,
                key_metadata,
                aad_file_unique,
                requires_aad_prefix: self.requires_aad_prefix,
            },
        )
    }
}

pub(crate) fn fill_random(bytes: &mut [u8]) -> VortexResult<()> {
    getrandom::fill(bytes).map_err(|e| vortex_err!("Failed to generate random bytes: {e}"))
}

#[cfg(test)]
mod tests;

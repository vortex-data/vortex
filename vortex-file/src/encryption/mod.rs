// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Segment-level encryption of Vortex files.
//!
//! Encryption operates on whole segments: data segments, and the dtype, layout, statistics,
//! footer, and user metadata segments referenced by the postscript. The postscript itself stays in
//! plaintext, since it only records segment locations and the [`EncryptionSpec`] of each.
//!
//! Writers configure a [`FileEncryption`] with a footer key, and optionally keys for individual
//! fields. Keys are never written to the file. Each key's opaque key metadata is stored instead,
//! so that readers configured with a [`FileDecryption`] can resolve the key again through a
//! [`KeyRetriever`]. The [`envelope`] module builds both from a key management service.
//!
//! The footer is either encrypted, which also hides the schema, or stored in plaintext and signed
//! with the footer key, so that readers without encryption support can still read the fields that
//! are not encrypted. Fields without their own key are encrypted with the footer key, or stored in
//! plaintext.
//!
//! Segments are encrypted with AES-GCM. The additional authenticated data binds every segment to its
//! file, through random per-file bytes, and to its position in the file, so that segments cannot
//! be swapped without failing decryption. An optional AAD prefix additionally binds the file to an
//! application-defined identity, such as its path, so that a file cannot be replaced by another
//! file encrypted with the same key. The prefix is not stored in the file, so readers must supply
//! it.

mod cipher;
pub mod envelope;
mod resolver;

use std::collections::BTreeMap;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use async_trait::async_trait;
use flatbuffers::FlatBufferBuilder;
use flatbuffers::WIPOffset;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldPath;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
pub use vortex_layout::segments::EncryptionKeyName;
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
/// footer is read, unless the reader opts out with
/// [`FileDecryption::without_field_key_statistics`], and other keys when a segment encrypted with
/// them is first read.
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
    /// the data keys of [`envelope::EnvelopeEncryption`], over one key shared by many files.
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
    plaintext_footer: bool,
    plaintext_fields: bool,
    aad_prefix: Option<Arc<[u8]>>,
    named_keys: BTreeMap<EncryptionKeyName, KeyConfig>,
    field_keys: Vec<(FieldPath, EncryptionKeyName)>,
}

impl FileEncryption {
    /// Encrypt the file with `footer_key`, which encrypts or signs the footer, and encrypts every
    /// field that has no key of its own.
    pub fn new(footer_key: KeyConfig) -> Self {
        Self {
            footer_key,
            algorithm: EncryptionAlgorithm::AesGcm,
            plaintext_footer: false,
            plaintext_fields: false,
            aad_prefix: None,
            named_keys: BTreeMap::new(),
            field_keys: Vec::new(),
        }
    }

    /// Set the encryption algorithm. Defaults to, and currently only supports,
    /// [`EncryptionAlgorithm::AesGcm`].
    pub fn with_algorithm(mut self, algorithm: EncryptionAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Store the footer, layout, dtype, and statistics in plaintext, signed with the footer key.
    ///
    /// Readers without encryption support, or without the footer key, can then read the fields
    /// that are stored in plaintext, and readers with the footer key verify the signature. The
    /// schema and layout tree are visible to anyone, as are the file statistics of plaintext
    /// fields; those of encrypted fields are encrypted with the field's key. User metadata segments
    /// stay encrypted.
    ///
    /// The layout tree reveals the row count, and the number and sizes of the segments, of every
    /// field, encrypted ones included, which can hint at their cardinality or compressibility.
    /// The footer also lists every array encoding used in the file, including those of encrypted
    /// fields.
    pub fn with_plaintext_footer(mut self) -> Self {
        self.plaintext_footer = true;
        self
    }

    /// Store fields without a key of their own in plaintext instead of encrypting them with the
    /// footer key.
    pub fn with_plaintext_fields(mut self) -> Self {
        self.plaintext_fields = true;
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

    /// Register a named key, replacing any key registered under the same name, for segments that
    /// a layout strategy requires to be encrypted with it, e.g. through
    /// [`WriteStrategyBuilder::with_field_encryption`](crate::WriteStrategyBuilder::with_field_encryption).
    pub fn with_named_key(mut self, name: impl Into<EncryptionKeyName>, key: KeyConfig) -> Self {
        self.named_keys.insert(name.into(), key);
        self
    }

    /// Encrypt a field, including its statistics and nested fields, with its own key when writing
    /// with the default layout strategy.
    ///
    /// This registers `key` under `name`, as [`with_named_key`](Self::with_named_key) does. A
    /// reader without the key can still read every other field, and does not see the file-level
    /// statistics of this one, which are encrypted with its key.
    ///
    /// With a custom strategy configure the field on the strategy instead; the write fails
    /// otherwise. The write also fails if `field` is the root, contains anything but field names,
    /// or overlaps another field passed to this method.
    pub fn with_field_key(
        mut self,
        field: impl Into<FieldPath>,
        name: impl Into<EncryptionKeyName>,
        key: KeyConfig,
    ) -> Self {
        let name = name.into();
        self.field_keys.push((field.into(), name.clone()));
        self.with_named_key(name, key)
    }

    /// The footer key.
    pub(crate) fn footer_key(&self) -> &KeyConfig {
        &self.footer_key
    }

    /// The index of the encryption spec whose readers can read the footer: the plaintext spec's
    /// for a plaintext footer, which anyone can read, and the footer key's otherwise.
    pub(crate) fn footer_spec_index(&self) -> u16 {
        if self.plaintext_footer { 0 } else { 1 }
    }

    /// Validate the field keys and key metadata before a file is written.
    pub(crate) fn validate(&self) -> VortexResult<()> {
        for (index, (path, _)) in self.field_keys.iter().enumerate() {
            if path.is_root() {
                vortex_bail!("Cannot encrypt the root field with a field key");
            }
            if !path
                .parts()
                .iter()
                .all(|part| matches!(part, Field::Name(_)))
            {
                vortex_bail!("Encrypted field path {path} must consist of field names");
            }
            if let Some((other, _)) = self.field_keys[..index]
                .iter()
                .find(|(other, _)| other.overlap(path))
            {
                vortex_bail!("Field keys of {path} and {other} overlap");
            }
        }

        // Readers resolve keys by their metadata, so it must identify a single key.
        let keys: Vec<&KeyConfig> = std::iter::once(&self.footer_key)
            .chain(self.named_keys.values())
            .collect();
        for (index, key) in keys.iter().enumerate() {
            if keys[..index].iter().any(|other| {
                other.key_metadata() == key.key_metadata()
                    && other.key().as_bytes() != key.key().as_bytes()
            }) {
                vortex_bail!("Different encryption keys share the same key metadata");
            }
        }
        Ok(())
    }

    /// Whether the footer is stored in plaintext and signed.
    pub fn is_plaintext_footer(&self) -> bool {
        self.plaintext_footer
    }

    /// Whether fields without a key of their own are stored in plaintext.
    pub fn is_plaintext_fields(&self) -> bool {
        self.plaintext_fields
    }

    pub(crate) fn named_keys(&self) -> &BTreeMap<EncryptionKeyName, KeyConfig> {
        &self.named_keys
    }

    pub(crate) fn field_keys(&self) -> &[(FieldPath, EncryptionKeyName)] {
        &self.field_keys
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
    pub(crate) verify_footer: bool,
    pub(crate) allow_plaintext_files: bool,
    pub(crate) cache_plaintext: bool,
    pub(crate) field_key_statistics: bool,
}

impl FileDecryption {
    /// Decrypt files with the keys resolved by `keys`.
    pub fn new(keys: Arc<dyn KeyRetriever>) -> Self {
        Self {
            keys,
            aad_prefix: None,
            verify_footer: true,
            allow_plaintext_files: false,
            cache_plaintext: false,
            field_key_statistics: true,
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

    /// Do not verify the signature of a plaintext footer.
    ///
    /// By default, the footer key of a file written with [`FileEncryption::with_plaintext_footer`]
    /// is retrieved to verify that the footer was not modified.
    ///
    /// This removes the integrity protection of the whole file, not only of its schema and
    /// statistics: the footer lists every segment and the encryption spec of each, so an
    /// unverified footer can mark encrypted segments as plaintext, or point them at other keys
    /// and at data of an attacker's choice. Decryption still keeps encrypted segments secret.
    pub fn without_footer_verification(mut self) -> Self {
        self.verify_footer = false;
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

    /// Do not retrieve field keys when the file is opened, to read the file statistics of fields
    /// encrypted with their own key.
    ///
    /// By default, opening a file retrieves the key of every field whose statistics are encrypted
    /// with it, even if no scan reads the field, so that its statistics are available for pruning.
    /// With a key management service, this calls the service for every such key on every open,
    /// and fails, unseen, for keys the reader may not use. With this option, field keys are only
    /// retrieved when a field's segments are read, and the statistics of such fields are empty.
    pub fn without_field_key_statistics(mut self) -> Self {
        self.field_key_statistics = false;
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
            .field("verify_footer", &self.verify_footer)
            .field("allow_plaintext_files", &self.allow_plaintext_files)
            .field("cache_plaintext", &self.cache_plaintext)
            .field("field_key_statistics", &self.field_key_statistics)
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

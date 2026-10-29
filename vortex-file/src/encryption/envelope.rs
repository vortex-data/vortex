// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Envelope encryption with a key management service (KMS).
//!
//! Master keys never leave the KMS. Every [`EnvelopeEncryption::key_config`] generates a random
//! data encryption key (DEK) and stores it, wrapped with a master key, as the key metadata of
//! the file. The [`EnvelopeEncryption`] is also the [`KeyRetriever`] that unwraps it again.
//!
//! With double wrapping, the default, DEKs are wrapped locally with a key encryption key (KEK),
//! and only the KEK is wrapped by the KMS. KEKs are cached per master key when writing, and by
//! id when reading, so that the KMS is called once per master key rather than once per file.
//!
//! Key material is stored in the file by default. With an external [`KeyMaterialStore`], the
//! file only references it, so that [`EnvelopeEncryption::rotate_master_keys`] can rewrap every
//! DEK with the current version of its master key without rewriting files.
//!
//! Cached KEKs expire after [`EnvelopeEncryption::with_kek_lifetime`], 10 minutes by default, so
//! that new files pick up master key rotations and revoked KMS access, rather than using a KEK for
//! the lifetime of the process.
//!
//! # Trust
//!
//! Readers trust every key the KMS unwraps for them. Anyone allowed to wrap keys with the footer
//! master key can therefore create a data key they know and write files that readers accept as
//! genuine. Integrity relies on restricting the permission to wrap with a master key as well as
//! the permission to unwrap with it.
//!
//! ```
//! # use std::sync::Arc;
//! # use vortex_array::dtype::FieldPath;
//! # use vortex_error::VortexResult;
//! # use vortex_file::encryption::{FileDecryption, FileEncryption};
//! # use vortex_file::encryption::envelope::{EnvelopeEncryption, KmsClient};
//! # async fn example(kms: Arc<dyn KmsClient>) -> VortexResult<()> {
//! let envelope = Arc::new(EnvelopeEncryption::new(kms));
//! let encryption = FileEncryption::new(envelope.key_config("footer-master-key").await?)
//!     .with_field_key(
//!         FieldPath::from_name("ssn"),
//!         "pii",
//!         envelope.key_config("pii-master-key").await?,
//!     );
//! let decryption = FileDecryption::new(envelope);
//! # Ok(())
//! # }
//! ```

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use async_trait::async_trait;
use flatbuffers::FlatBufferBuilder;
use flatbuffers::WIPOffset;
use parking_lot::Mutex;
use parking_lot::RwLock;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;
use zeroize::Zeroize;

use super::Aead;
use super::EncryptionKey;
use super::KeyConfig;
use super::KeyRetriever;
use super::fill_random;
use crate::flatbuffers::key_material as fb;

const KEK_LEN: usize = 32;
const KEK_ID_LEN: usize = 16;
const EXTERNAL_ID_LEN: usize = 16;
const DEFAULT_KEK_LIFETIME: Duration = Duration::from_secs(10 * 60);

/// A client of a key management service that holds master keys.
///
/// Implementations wrap and unwrap keys with a remote service, such as AWS KMS, Google Cloud KMS,
/// or HashiCorp Vault, and should support master key rotation by unwrapping keys wrapped with any
/// version of a master key.
#[async_trait]
pub trait KmsClient: Send + Sync {
    /// Wrap `key` with the current version of the master key `master_key_id`.
    async fn wrap_key(&self, key: &[u8], master_key_id: &str) -> VortexResult<Vec<u8>>;

    /// Unwrap a key wrapped with any version of the master key `master_key_id`.
    async fn unwrap_key(&self, wrapped_key: &[u8], master_key_id: &str) -> VortexResult<Vec<u8>>;
}

/// Stores key material outside of the files it belongs to.
///
/// [`EnvelopeEncryption::key_config`] stores key material before any file uses it, so material
/// of files that were never written, or were deleted, stays in the store until its owner removes
/// it.
#[async_trait]
pub trait KeyMaterialStore: Send + Sync {
    /// Store, or replace, the key material with `id`.
    async fn put(&self, id: &str, material: &[u8]) -> VortexResult<()>;

    /// Return the key material with `id`.
    async fn get(&self, id: &str) -> VortexResult<Vec<u8>>;

    /// Return the ids of all stored key material.
    async fn ids(&self) -> VortexResult<Vec<String>>;
}

/// A [`KeyMaterialStore`] in memory.
#[derive(Default)]
pub struct InMemoryKeyMaterialStore {
    materials: RwLock<HashMap<String, Vec<u8>>>,
}

#[async_trait]
impl KeyMaterialStore for InMemoryKeyMaterialStore {
    async fn put(&self, id: &str, material: &[u8]) -> VortexResult<()> {
        self.materials
            .write()
            .insert(id.to_string(), material.to_vec());
        Ok(())
    }

    async fn get(&self, id: &str) -> VortexResult<Vec<u8>> {
        self.materials
            .read()
            .get(id)
            .cloned()
            .ok_or_else(|| vortex_err!("No key material with id {id}"))
    }

    async fn ids(&self) -> VortexResult<Vec<String>> {
        Ok(self.materials.read().keys().cloned().collect())
    }
}

/// A key encryption key used to wrap data keys locally.
struct Kek {
    id: Vec<u8>,
    key: EncryptionKey,
    /// The KEK wrapped by the KMS, when writing.
    wrapped: Vec<u8>,
    created: Option<Instant>,
}

/// A KEK unwrapped when reading.
#[derive(Clone)]
struct CachedKek {
    key: EncryptionKey,
    created: Option<Instant>,
}

/// The outcome of [`EnvelopeEncryption::rotate_master_keys`].
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct RotationReport {
    /// The number of data keys rewrapped.
    pub rewrapped: usize,
    /// The ids of the key material that could not be rewrapped, and why. Files that use them
    /// stay readable as long as the KMS can unwrap their previous wrapping.
    pub failed: Vec<(String, VortexError)>,
}

/// Generates data keys wrapped by a [`KmsClient`], and unwraps them as a [`KeyRetriever`].
pub struct EnvelopeEncryption {
    kms: Arc<dyn KmsClient>,
    double_wrapping: bool,
    data_key_length: usize,
    kek_lifetime: Duration,
    store: Option<Arc<dyn KeyMaterialStore>>,
    /// KEKs used to wrap new data keys, by master key id.
    write_keks: Mutex<HashMap<String, Arc<Kek>>>,
    /// Unwrapped KEKs, by the master key that wraps them and their KEK id.
    ///
    /// Key material comes from files, which may have been tampered with. Keying the cache by the
    /// master key too means that material claiming a cached KEK under another master key misses
    /// the cache, and the KMS decides whether the reader may use that master key.
    read_keks: Mutex<HashMap<KekCacheKey, CachedKek>>,
}

/// A master key id and a KEK id.
type KekCacheKey = (String, Vec<u8>);

impl EnvelopeEncryption {
    /// Wrap keys with master keys held by `kms`, with double wrapping and 128-bit data keys by
    /// default.
    pub fn new(kms: Arc<dyn KmsClient>) -> Self {
        Self {
            kms,
            double_wrapping: true,
            data_key_length: 16,
            kek_lifetime: DEFAULT_KEK_LIFETIME,
            store: None,
            write_keks: Mutex::default(),
            read_keks: Mutex::default(),
        }
    }

    /// Wrap every data key with the KMS directly, rather than with a cached key encryption key.
    ///
    /// This calls the KMS once for every key written and read.
    pub fn with_single_wrapping(mut self) -> Self {
        self.double_wrapping = false;
        self
    }

    /// Set the length in bytes of generated data keys: 16, 24, or 32.
    pub fn with_data_key_length(mut self, data_key_length: usize) -> VortexResult<Self> {
        if !super::KEY_LENGTHS.contains(&data_key_length) {
            vortex_bail!("Data keys must be 16, 24, or 32 bytes, but got {data_key_length} bytes");
        }
        self.data_key_length = data_key_length;
        Ok(self)
    }

    /// Set how long a cached key encryption key is used, 10 minutes by default.
    ///
    /// A new KEK, wrapped with the current version of its master key, replaces an expired one
    /// when writing, and an expired KEK is unwrapped again by the KMS when reading. On `wasm32`,
    /// which has no clock, cached KEKs do not expire; use
    /// [`clear_caches`](Self::clear_caches) instead.
    pub fn with_kek_lifetime(mut self, kek_lifetime: Duration) -> Self {
        self.kek_lifetime = kek_lifetime;
        self
    }

    /// Keep key material in `store`, and only reference it from files.
    ///
    /// Readers must use an [`EnvelopeEncryption`] with the same store.
    pub fn with_external_key_material(mut self, store: Arc<dyn KeyMaterialStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Generate a fresh data key wrapped with the master key `master_key_id`.
    ///
    /// Use a fresh key for every file, so that no key encrypts more data than AES-GCM's random
    /// nonces allow.
    pub async fn key_config(&self, master_key_id: &str) -> VortexResult<KeyConfig> {
        let dek = EncryptionKey::generate(self.data_key_length)?;
        let wrapping = self.wrap_dek(&dek, master_key_id).await?;
        let mut fbb = FlatBufferBuilder::new();
        let args = match &self.store {
            None => fb::EnvelopeKeyMetadataArgs {
                material: Some(write_material(&mut fbb, master_key_id, &wrapping)),
                external_id: None,
            },
            Some(store) => {
                let mut id = [0u8; EXTERNAL_ID_LEN];
                fill_random(&mut id)?;
                let id = hex(&id);
                store
                    .put(&id, &material_bytes(master_key_id, &wrapping))
                    .await?;
                fb::EnvelopeKeyMetadataArgs {
                    material: None,
                    external_id: Some(fbb.create_string(&id)),
                }
            }
        };
        let metadata = fb::EnvelopeKeyMetadata::create(&mut fbb, &args);
        fbb.finish_minimal(metadata);
        Ok(KeyConfig::new(dek, fbb.finished_data()))
    }

    /// Rewrap every data key in the external key material store with the current version of its
    /// master key.
    ///
    /// The [`KmsClient`] must be able to unwrap keys wrapped with earlier master key versions.
    /// Files keep referencing the same key material, so they stay readable. Key material that
    /// fails to rewrap is reported and left as it was, and the rotation continues with the rest,
    /// so a failed rotation can be retried.
    ///
    /// Each entry is replaced on its own, not atomically with the others. Key material stored by
    /// [`key_config`](Self::key_config) while the rotation runs may be missed, but is wrapped with
    /// a KEK created after the rotation started.
    pub async fn rotate_master_keys(&self) -> VortexResult<RotationReport> {
        let Some(store) = &self.store else {
            vortex_bail!(
                "Master key rotation requires external key material, configured with \
                 EnvelopeEncryption::with_external_key_material"
            );
        };
        // New KEKs, wrapped with the current master key versions, replace the cached ones.
        self.write_keks.lock().clear();
        let mut report = RotationReport::default();
        for id in store.ids().await? {
            match self.rewrap(store.as_ref(), &id).await {
                Ok(()) => report.rewrapped += 1,
                Err(err) => report.failed.push((id, err)),
            }
        }
        Ok(report)
    }

    /// Rewrap the data key of the key material `id` with the current version of its master key.
    async fn rewrap(&self, store: &dyn KeyMaterialStore, id: &str) -> VortexResult<()> {
        let material = Material::parse(&store.get(id).await?)?;
        let dek = self.unwrap(&material).await?;
        let wrapping = self.wrap_dek(&dek, &material.master_key_id).await?;
        store
            .put(id, &material_bytes(&material.master_key_id, &wrapping))
            .await
    }

    /// Drop the cached key encryption keys, e.g. when KMS access tokens are revoked.
    pub fn clear_caches(&self) {
        self.write_keks.lock().clear();
        self.read_keks.lock().clear();
    }

    /// Whether a KEK cached at `created` has expired.
    fn expired(&self, created: Option<Instant>) -> bool {
        created.is_some_and(|created| created.elapsed() >= self.kek_lifetime)
    }

    /// Wrap `dek` with the master key `master_key_id`, through a KEK with double wrapping.
    async fn wrap_dek(&self, dek: &EncryptionKey, master_key_id: &str) -> VortexResult<Wrapping> {
        if !self.double_wrapping {
            return Ok(Wrapping {
                wrapped_dek: self.kms.wrap_key(dek.as_bytes(), master_key_id).await?,
                kek: None,
            });
        }
        let kek = self.write_kek(master_key_id).await?;
        let mut wrapped_dek = dek.as_bytes().to_vec();
        let nonce_tag = Aead::new(&kek.key)?.seal(&kek.id, &mut wrapped_dek)?;
        wrapped_dek.extend_from_slice(&nonce_tag);
        Ok(Wrapping {
            wrapped_dek,
            kek: Some(kek),
        })
    }

    /// The KEK for new data keys wrapped with `master_key_id`, created on first use and when the
    /// cached one expires.
    async fn write_kek(&self, master_key_id: &str) -> VortexResult<Arc<Kek>> {
        if let Some(kek) = self.write_keks.lock().get(master_key_id)
            && !self.expired(kek.created)
        {
            return Ok(Arc::clone(kek));
        }
        let key = EncryptionKey::generate(KEK_LEN)?;
        let mut id = vec![0u8; KEK_ID_LEN];
        fill_random(&mut id)?;
        let wrapped = self.kms.wrap_key(key.as_bytes(), master_key_id).await?;
        let created = now();
        self.read_keks.lock().insert(
            (master_key_id.to_string(), id.clone()),
            CachedKek {
                key: key.clone(),
                created,
            },
        );
        let kek = Arc::new(Kek {
            id,
            key,
            wrapped,
            created,
        });
        // A concurrent call may have created a fresh KEK meanwhile; keep that one.
        let mut write_keks = self.write_keks.lock();
        let cached = write_keks
            .entry(master_key_id.to_string())
            .or_insert_with(|| Arc::clone(&kek));
        if self.expired(cached.created) {
            *cached = kek;
        }
        Ok(Arc::clone(cached))
    }

    /// Unwrap the data key of `material`.
    async fn unwrap(&self, material: &Material) -> VortexResult<EncryptionKey> {
        let Some((kek_id, wrapped_kek)) = &material.kek else {
            let mut dek = self
                .kms
                .unwrap_key(&material.wrapped_dek, &material.master_key_id)
                .await?;
            let key = EncryptionKey::try_from_slice(&dek);
            dek.zeroize();
            return key;
        };
        let cache_key = (material.master_key_id.clone(), kek_id.clone());
        let cached = self
            .read_keks
            .lock()
            .get(&cache_key)
            .filter(|kek| !self.expired(kek.created))
            .map(|kek| kek.key.clone());
        let kek = match cached {
            Some(kek) => kek,
            None => {
                let mut kek = self
                    .kms
                    .unwrap_key(wrapped_kek, &material.master_key_id)
                    .await?;
                let key = EncryptionKey::try_from_slice(&kek);
                kek.zeroize();
                let key = key?;
                self.read_keks.lock().insert(
                    cache_key,
                    CachedKek {
                        key: key.clone(),
                        created: now(),
                    },
                );
                key
            }
        };
        let wrapped_dek = &material.wrapped_dek;
        let Some(len) = wrapped_dek.len().checked_sub(super::NONCE_TAG_LEN) else {
            vortex_bail!(
                "Wrapped data key of {} bytes is too short",
                wrapped_dek.len()
            );
        };
        let mut dek = wrapped_dek[..len].to_vec();
        let opened = Aead::new(&kek)?.open(kek_id, &mut dek, &wrapped_dek[len..]);
        let key = opened
            .map_err(|()| vortex_err!("Failed to unwrap the data key with its key encryption key"))
            .and_then(|()| EncryptionKey::try_from_slice(&dek));
        dek.zeroize();
        key
    }
}

/// A data key wrapped for key material.
struct Wrapping {
    wrapped_dek: Vec<u8>,
    /// With double wrapping, the KEK the data key is wrapped with.
    kek: Option<Arc<Kek>>,
}

/// Parsed key material, owned so that it can be held across KMS calls.
struct Material {
    master_key_id: String,
    wrapped_dek: Vec<u8>,
    /// With double wrapping, the KEK id and the KEK wrapped by the KMS.
    kek: Option<(Vec<u8>, Vec<u8>)>,
}

impl Material {
    fn parse(bytes: &[u8]) -> VortexResult<Self> {
        Self::from_flatbuffer(
            &flatbuffers::root::<fb::KeyMaterial>(bytes)
                .map_err(|e| vortex_err!("Invalid key material: {e}"))?,
        )
    }

    fn from_flatbuffer(material: &fb::KeyMaterial<'_>) -> VortexResult<Self> {
        let kek = match (material.kek_id(), material.wrapped_kek()) {
            (Some(id), Some(wrapped)) => Some((id.bytes().to_vec(), wrapped.bytes().to_vec())),
            (None, None) => None,
            _ => vortex_bail!("Key material must hold both or neither of a KEK id and wrapped KEK"),
        };
        Ok(Self {
            master_key_id: material.master_key_id().to_string(),
            wrapped_dek: material.wrapped_dek().bytes().to_vec(),
            kek,
        })
    }
}

/// Write the key material of `wrapping` to `fbb`.
fn write_material<'fb>(
    fbb: &mut FlatBufferBuilder<'fb>,
    master_key_id: &str,
    wrapping: &Wrapping,
) -> WIPOffset<fb::KeyMaterial<'fb>> {
    let args = fb::KeyMaterialArgs {
        master_key_id: Some(fbb.create_string(master_key_id)),
        wrapped_dek: Some(fbb.create_vector(&wrapping.wrapped_dek)),
        kek_id: wrapping.kek.as_ref().map(|kek| fbb.create_vector(&kek.id)),
        wrapped_kek: wrapping
            .kek
            .as_ref()
            .map(|kek| fbb.create_vector(&kek.wrapped)),
    };
    fb::KeyMaterial::create(fbb, &args)
}

/// Serialize the key material of `wrapping`.
fn material_bytes(master_key_id: &str, wrapping: &Wrapping) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::new();
    let material = write_material(&mut fbb, master_key_id, wrapping);
    fbb.finish_minimal(material);
    fbb.finished_data().to_vec()
}

#[async_trait]
impl KeyRetriever for EnvelopeEncryption {
    async fn retrieve_key(&self, key_metadata: &[u8]) -> VortexResult<EncryptionKey> {
        let metadata = flatbuffers::root::<fb::EnvelopeKeyMetadata>(key_metadata)
            .map_err(|e| vortex_err!("Invalid envelope key metadata: {e}"))?;
        let material = match (metadata.material(), metadata.external_id()) {
            (Some(material), None) => Material::from_flatbuffer(&material)?,
            (None, Some(id)) => {
                let Some(store) = &self.store else {
                    vortex_bail!(
                        "The key material of the file is external, but no key material store was \
                         configured with EnvelopeEncryption::with_external_key_material"
                    );
                };
                Material::parse(&store.get(id).await?)?
            }
            _ => vortex_bail!("Envelope key metadata must hold either key material or its id"),
        };
        self.unwrap(&material).await
    }
}

impl Debug for EnvelopeEncryption {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("EnvelopeEncryption")
            .field("double_wrapping", &self.double_wrapping)
            .field("data_key_length", &self.data_key_length)
            .field("external_key_material", &self.store.is_some())
            .finish_non_exhaustive()
    }
}

/// The current time, or `None` on `wasm32`, where `Instant::now` panics.
fn now() -> Option<Instant> {
    if cfg!(target_arch = "wasm32") {
        None
    } else {
        Some(Instant::now())
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

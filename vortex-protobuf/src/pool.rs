// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A process-wide cache of descriptor pools, keyed by the bytes of their descriptor sets.
//!
//! Protobuf metadata is deserialized whenever a file opens or an Arrow field imports, and each
//! time it would parse its descriptor set into a new pool. The descriptor set of a schema is
//! canonical, so the same schema always has the same bytes, and the cache parses them once. Two
//! metadata values of one schema then share the pool, and their message descriptors are equal.

use std::sync::Arc;
use std::sync::LazyLock;

use parking_lot::Mutex;
use prost::Message;
use prost_reflect::Cardinality;
use prost_reflect::DescriptorPool;
use prost_reflect::Kind;
use prost_reflect::MessageDescriptor;
use prost_types::FileDescriptorSet;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

/// The number of descriptor sets that the cache keeps. When it is full, the one used least
/// recently is dropped.
pub(crate) const CAPACITY: usize = 64;

/// The cached pools.
static CACHE: LazyLock<Mutex<PoolCache>> = LazyLock::new(|| Mutex::new(PoolCache::default()));

/// Returns the pool parsed from the serialized `descriptor_set`, and a shared copy of the bytes.
///
/// # Errors
///
/// Returns an error if the bytes are not a valid descriptor set of proto2 and proto3 files, or if
/// a map entry message does not have the fields of one.
pub(crate) fn descriptor_pool(descriptor_set: &[u8]) -> VortexResult<(Arc<[u8]>, DescriptorPool)> {
    if let Some(cached) = CACHE.lock().get(descriptor_set) {
        return Ok(cached);
    }
    // Parse without holding the lock. If another thread parses the same bytes meanwhile, the
    // pool that is inserted first wins.
    let files = FileDescriptorSet::decode(descriptor_set)
        .map_err(|err| vortex_err!("Invalid protobuf descriptor set: {err}"))?;
    // `prost-reflect` panics on a syntax that it does not know, such as `editions`, instead of
    // returning an error.
    if let Some(file) = files
        .file
        .iter()
        .find(|file| !matches!(file.syntax.as_deref(), None | Some("proto2" | "proto3")))
    {
        vortex_bail!(
            "Invalid protobuf descriptor set: file {} has syntax {:?}, but only proto2 and \
             proto3 are supported",
            file.name(),
            file.syntax()
        );
    }
    let pool = DescriptorPool::from_file_descriptor_set(files)
        .map_err(|err| vortex_err!("Invalid protobuf descriptor set: {err}"))?;
    validate_map_entries(&pool)?;
    Ok(CACHE.lock().insert(Arc::from(descriptor_set), pool))
}

/// Checks that every map entry message of `pool` has the fields that `protoc` generates for a
/// `map<K, V>` field: a singular key field 1 of an integer, bool or string type, and a singular
/// value field 2.
///
/// `prost-reflect` does not check map entry messages, and panics when it looks up the key or value
/// field of one that lacks it.
fn validate_map_entries(pool: &DescriptorPool) -> VortexResult<()> {
    for entry in pool.all_messages().filter(MessageDescriptor::is_map_entry) {
        let singular = |number| {
            entry
                .get_field(number)
                .filter(|field| field.cardinality() != Cardinality::Repeated)
        };
        let (Some(key), Some(_), 2) = (singular(1), singular(2), entry.fields().len()) else {
            vortex_bail!(
                "Invalid protobuf descriptor set: map entry message {} must have a singular key \
                 field 1 and a singular value field 2, and no other fields",
                entry.full_name()
            );
        };
        if matches!(
            key.kind(),
            Kind::Double | Kind::Float | Kind::Bytes | Kind::Enum(_) | Kind::Message(_)
        ) {
            vortex_bail!(
                "Invalid protobuf descriptor set: map entry message {} has a key that is not an \
                 integer, bool or string",
                entry.full_name()
            );
        }
    }
    Ok(())
}

/// Descriptor pools by the bytes of their descriptor sets, with a clock for their last use.
#[derive(Default)]
pub(crate) struct PoolCache {
    /// The pools, with the time of their last use.
    pools: HashMap<Arc<[u8]>, (DescriptorPool, u64)>,
    /// Counts the lookups and insertions.
    clock: u64,
}

impl PoolCache {
    /// Returns the pool of `descriptor_set` and the cached bytes, if the cache holds them.
    pub(crate) fn get(&mut self, descriptor_set: &[u8]) -> Option<(Arc<[u8]>, DescriptorPool)> {
        self.clock += 1;
        let clock = self.clock;
        let (bytes, (pool, last_used)) = self.pools.get_key_value_mut(descriptor_set)?;
        *last_used = clock;
        Some((Arc::clone(bytes), pool.clone()))
    }

    /// Caches `pool` as the pool of `descriptor_set`, unless the cache holds one already, and
    /// returns the cached pool and bytes.
    pub(crate) fn insert(
        &mut self,
        descriptor_set: Arc<[u8]>,
        pool: DescriptorPool,
    ) -> (Arc<[u8]>, DescriptorPool) {
        if let Some(cached) = self.get(&descriptor_set) {
            return cached;
        }
        if self.pools.len() >= CAPACITY
            && let Some(oldest) = self
                .pools
                .iter()
                .min_by_key(|(_, (_, last_used))| *last_used)
                .map(|(bytes, _)| Arc::clone(bytes))
        {
            self.pools.remove(&oldest);
        }
        self.clock += 1;
        self.pools
            .insert(Arc::clone(&descriptor_set), (pool.clone(), self.clock));
        (descriptor_set, pool)
    }
}

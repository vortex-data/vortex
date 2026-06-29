// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheduler-visible scan read requests.
//!
//! These types intentionally do not model layout segments. Layouts and file readers
//! decide what a read key means; the scan scheduler only needs a stable key, byte
//! estimate, phase, priority, and cancellation scope for admission and deduplication.

use std::sync::Arc;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

/// Static future resolving to a scan read buffer.
pub type ScanReadFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// High-level scan phase associated with a read request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ScanIoPhase {
    /// Shared evidence setup, such as loading a stats table.
    EvidenceSetup,
    /// Per-morsel evidence probe.
    EvidenceProbe,
    /// Residual predicate value read.
    PredicateRead,
    /// Projected output value read.
    #[default]
    ProjectionRead,
    /// Aggregate input or metadata read.
    AggregateRead,
}

/// Scheduler priority for a read request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScanPriority(i32);

impl ScanPriority {
    /// Normal request priority.
    pub const NORMAL: Self = Self(0);

    /// Create a priority from a signed integer value.
    pub fn new(value: i32) -> Self {
        Self(value)
    }

    /// Return the signed integer priority value.
    pub fn get(self) -> i32 {
        self.0
    }
}

/// Cancellation scope for a group of related read requests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CancelGroup(u64);

impl CancelGroup {
    /// A request that is not associated with a finer cancellation group.
    pub const NONE: Self = Self(0);

    /// Create a cancellation group from an integer id.
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the integer cancellation group id.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Opaque dedupe key for a logical scan read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReadRequestKey(u64);

impl ReadRequestKey {
    /// Create an opaque read request key.
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the raw key value.
    pub fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for ReadRequestKey {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

/// A scheduler-visible request for one logical read payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScanReadRequest {
    /// Opaque read dedupe key.
    pub key: ReadRequestKey,
    /// Number of logical bytes this read contributes to admission.
    pub bytes: u64,
    /// High-level scan phase that needs this read.
    pub phase: ScanIoPhase,
    /// Scheduler priority for this request.
    pub priority: ScanPriority,
    /// Cancellation scope for this request.
    pub cancel_group: CancelGroup,
}

impl ScanReadRequest {
    /// Create a read request with normal priority and no cancellation group.
    pub fn new(key: ReadRequestKey, bytes: u64, phase: ScanIoPhase) -> Self {
        Self {
            key,
            bytes,
            phase,
            priority: ScanPriority::NORMAL,
            cancel_group: CancelGroup::NONE,
        }
    }

    /// Return a copy of this request with the provided priority.
    pub fn with_priority(mut self, priority: ScanPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Return a copy of this request with the provided cancellation group.
    pub fn with_cancel_group(mut self, cancel_group: CancelGroup) -> Self {
        self.cancel_group = cancel_group;
        self
    }
}

/// One logical read registered for a scan task.
pub struct ScanRead {
    /// The logical request this handle resolves.
    pub request: ScanReadRequest,
    /// Future resolving to the requested payload.
    pub future: ScanReadFuture,
}

/// Scan-wide store of resolved read buffers.
#[derive(Default)]
pub struct ReadStore {
    entries: Mutex<HashMap<ReadRequestKey, BufferHandle>>,
}

impl ReadStore {
    /// Create an empty read store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a resolved buffer by key, if present.
    pub fn get(&self, key: ReadRequestKey) -> Option<BufferHandle> {
        self.entries.lock().get(&key).cloned()
    }

    /// Insert a resolved buffer.
    pub fn insert(&self, key: ReadRequestKey, buffer: BufferHandle) {
        self.entries.lock().insert(key, buffer);
    }

    /// Remove a resolved buffer.
    pub fn remove(&self, key: ReadRequestKey) -> Option<BufferHandle> {
        self.entries.lock().remove(&key)
    }

    /// Remove multiple resolved buffers.
    pub fn remove_many(&self, keys: impl IntoIterator<Item = ReadRequestKey>) {
        let mut entries = self.entries.lock();
        for key in keys {
            entries.remove(&key);
        }
    }
}

/// Shared scan-wide read store.
pub type ReadStoreRef = Arc<ReadStore>;

/// Read-only view over resolved scan reads.
#[derive(Clone)]
pub struct ReadResults {
    store: ReadStoreRef,
}

impl ReadResults {
    /// Create a read results view over a shared store.
    pub fn new(store: ReadStoreRef) -> Self {
        Self { store }
    }

    /// Return a resolved buffer by key.
    pub fn get(&self, key: ReadRequestKey) -> VortexResult<BufferHandle> {
        self.store
            .get(key)
            .ok_or_else(|| vortex_err!("scan read {:?} was not resolved before execution", key))
    }

    /// Return whether a read has already been resolved.
    pub fn contains(&self, key: ReadRequestKey) -> bool {
        self.store.get(key).is_some()
    }

    /// Return the backing read store.
    pub fn store(&self) -> ReadStoreRef {
        Arc::clone(&self.store)
    }
}

impl ScanRead {
    /// Create a handle for one logical read request.
    pub fn new(request: ScanReadRequest, future: ScanReadFuture) -> Self {
        Self { request, future }
    }
}

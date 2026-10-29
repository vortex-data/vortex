// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::BuildHasher;
use std::sync::Arc;
use std::sync::Weak;

use parking_lot::Mutex;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_map::RandomState;

use super::Work;

const SHARDS: usize = 16;
type Key = (u64, usize, usize);

/// Coordinates identical pending mapped reads without retaining completed buffers.
#[derive(Default)]
pub(super) struct InflightReads {
    hash: RandomState,
    shards: [Mutex<HashMap<Key, Weak<PendingRead>>>; SHARDS],
}

pub(super) struct PendingRead {
    registry: Arc<InflightReads>,
    key: Key,
    waiters: Mutex<Option<Vec<Work>>>,
}

impl InflightReads {
    fn shard(&self, key: Key) -> &Mutex<HashMap<Key, Weak<PendingRead>>> {
        let shard = self.hash.hash_one(key) % SHARDS as u64;
        &self.shards[usize::try_from(shard).vortex_expect("Shard index fits usize")]
    }

    pub(super) fn join(self: &Arc<Self>, mut work: Work) -> Option<Work> {
        let Some(mapping) = work.mapping.as_ref() else {
            return Some(work);
        };
        // A file can acquire a new mapping while old scopes still use the previous one.
        let key = (work.offset, work.length, Arc::as_ptr(mapping) as usize);
        let mut entries = self.shard(key).lock();
        if let Some(pending) = entries.get(&key).and_then(Weak::upgrade) {
            let mut guard = pending.waiters.lock();
            if let Some(waiters) = guard.as_mut() {
                waiters.push(work);
                // Dropping the last lease can fail its waiters and remove its registry entry.
                drop(guard);
                drop(entries);
                return None;
            }
        }
        let pending = Arc::new(PendingRead {
            registry: Arc::clone(self),
            key,
            waiters: Mutex::new(Some(Vec::new())),
        });
        entries.insert(key, Arc::downgrade(&pending));
        work.inflight = Some(pending);
        Some(work)
    }

    #[cfg(test)]
    pub(super) fn pending_count(&self) -> usize {
        self.shards.iter().map(|shard| shard.lock().len()).sum()
    }
}

impl PendingRead {
    pub(super) fn complete(&self, result: &VortexResult<ByteBuffer>) {
        let Some(waiters) = self.waiters.lock().take() else {
            return;
        };
        let mut entries = self.registry.shard(self.key).lock();
        if entries
            .get(&self.key)
            .is_some_and(|entry| std::ptr::eq(entry.as_ptr(), self))
        {
            entries.remove(&self.key);
        }
        drop(entries);
        for work in waiters {
            if work.live() {
                let result = result.clone().map(|bytes| bytes.aligned(work.alignment));
                work.complete(result);
            }
        }
    }
}

impl Drop for PendingRead {
    fn drop(&mut self) {
        if self.waiters.get_mut().is_some() {
            self.complete(&Err(vortex_err!(
                "Owning ring dropped a pending shared read"
            )));
        }
    }
}

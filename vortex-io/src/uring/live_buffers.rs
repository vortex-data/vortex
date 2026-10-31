// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::BuildHasher;
use std::sync::Arc;
use std::sync::Weak;

use bytes::Bytes;
use parking_lot::Mutex;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_map::RandomState;

const SHARDS: usize = 16;
const ENTRIES_PER_SHARD: usize = 256;
type Key = (u64, usize);

/// Indexes live consumer buffers of one immutable file, without retaining their data.
#[derive(Default)]
pub(super) struct LiveBuffers {
    hash: RandomState,
    shards: [Mutex<HashMap<Key, Weak<Lease>>>; SHARDS],
}

struct Lease(ByteBuffer);

struct Owner(Arc<Lease>);

impl AsRef<[u8]> for Owner {
    fn as_ref(&self) -> &[u8] {
        self.0.0.as_slice()
    }
}

impl LiveBuffers {
    fn shard(&self, key: Key) -> &Mutex<HashMap<Key, Weak<Lease>>> {
        let shard = self.hash.hash_one(key) % SHARDS as u64;
        &self.shards[usize::try_from(shard).vortex_expect("Shard index fits usize")]
    }

    pub(super) fn get(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> Option<ByteBuffer> {
        let key = (offset, length);
        let lease = self.shard(key).lock().get(&key)?.upgrade()?;
        Some(Self::buffer(lease, alignment))
    }

    pub(super) fn insert(&self, offset: u64, bytes: ByteBuffer) -> ByteBuffer {
        let key = (offset, bytes.len());
        let alignment = bytes.alignment();
        let mut entries = self.shard(key).lock();
        if let Some(lease) = entries.get(&key).and_then(Weak::upgrade) {
            drop(entries);
            return Self::buffer(lease, alignment);
        }
        if entries.len() >= ENTRIES_PER_SHARD {
            entries.retain(|_, lease| lease.strong_count() != 0);
            if entries.len() >= ENTRIES_PER_SHARD {
                return bytes;
            }
        }
        let lease = Arc::new(Lease(bytes));
        entries.insert(key, Arc::downgrade(&lease));
        drop(entries);
        Self::buffer(lease, alignment)
    }

    fn buffer(lease: Arc<Lease>, alignment: Alignment) -> ByteBuffer {
        ByteBuffer::from(Bytes::from_owner(Owner(lease))).aligned(alignment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_keep_a_lease_alive_until_the_last_consumer_drops() {
        let registry = LiveBuffers::default();
        let original = registry.insert(8, ByteBuffer::from(vec![1, 2, 3, 4]));
        let slice = original.slice_unaligned(1..3);
        drop(original);
        let reused = registry
            .get(8, 4, Alignment::none())
            .expect("live slice retains the range");
        assert_eq!(reused.as_slice(), &[1, 2, 3, 4]);
        drop(slice);
        assert!(registry.get(8, 4, Alignment::none()).is_some());
        drop(reused);
        assert!(registry.get(8, 4, Alignment::none()).is_none());
    }

    #[test]
    fn ranges_and_files_are_independent_and_alignment_is_preserved() {
        let registry = LiveBuffers::default();
        let other = LiveBuffers::default();
        let bytes = registry.insert(8, ByteBuffer::from(vec![31; 65]).slice_unaligned(1..65));
        assert!(registry.get(9, 64, Alignment::none()).is_none());
        assert!(registry.get(8, 63, Alignment::none()).is_none());
        assert!(other.get(8, 64, Alignment::none()).is_none());
        let aligned = registry
            .get(8, 64, Alignment::new(64))
            .expect("live buffer");
        assert!(aligned.is_aligned(Alignment::new(64)));
        assert_eq!(aligned.as_slice(), bytes.as_slice());
    }

    #[test]
    fn metadata_is_bounded_even_while_all_consumers_remain_live() {
        let registry = LiveBuffers::default();
        let buffers: Vec<_> = (0..SHARDS * ENTRIES_PER_SHARD * 2)
            .map(|offset| registry.insert(offset as u64, ByteBuffer::from(vec![7])))
            .collect();
        assert!(
            registry
                .shards
                .iter()
                .all(|shard| shard.lock().len() <= ENTRIES_PER_SHARD)
        );
        assert!(buffers.iter().all(|bytes| bytes.as_slice() == [7]));
        drop(buffers);
        assert!(
            registry
                .shards
                .iter()
                .all(|shard| shard.lock().values().all(|lease| lease.upgrade().is_none()))
        );
        let replacement = registry.insert(0, ByteBuffer::from(vec![8]));
        assert_eq!(
            registry.get(0, 1, Alignment::none()).expect("new lease"),
            replacement
        );
    }
}

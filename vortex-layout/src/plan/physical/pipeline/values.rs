// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Whole values and decoded segments kept between the scans that share them.

use std::collections::VecDeque;
use std::hash::BuildHasher;
use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::FxBuildHasher;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;

use crate::plan::physical::PlanByPtr;
use crate::plan::physical::PlanRef;

/// Values a plan produces over its whole domain whatever rows it is compiled for, kept for the
/// later scans that share this cache.
///
/// A dictionary's values are read whole by every reader of its codes, so readers sharing this
/// cache can reuse them across filters, projections, and splits. The caller controls its lifetime;
/// only whole values that a reader requested are retained. V2 scans also use a bounded cache of
/// full encoded arrays to share segment deserialization between chunks.
#[derive(Clone, Default)]
pub struct ValueCache {
    values: Arc<Mutex<FxHashMap<PlanByPtr, ArrayRef>>>,
    decoded: Option<DecodedSegments>,
}

impl ValueCache {
    /// The values `plan` produced earlier.
    pub(crate) fn get(&self, plan: &PlanRef) -> Option<ArrayRef> {
        self.values.lock().get(&PlanByPtr(plan.clone())).cloned()
    }

    /// Records the values of `plan`, keeping the first when two race.
    pub(crate) fn insert(&self, plan: &PlanRef, values: ArrayRef) -> ArrayRef {
        self.values
            .lock()
            .entry(PlanByPtr(plan.clone()))
            .or_insert(values)
            .clone()
    }

    pub(crate) fn with_decoded_segments(capacity: u64) -> Self {
        Self {
            decoded: Some(DecodedSegments(Arc::new(DecodedCache {
                shards: Default::default(),
                capacity: capacity / DECODED_SHARDS as u64,
            }))),
            ..Self::default()
        }
    }

    pub(crate) fn decoded(&self) -> Option<DecodedSegments> {
        self.decoded.clone()
    }
}

/// Full encoded arrays shared by overlapping scans of one file. Entries are weighted by logical
/// buffer bytes plus an allowance for metadata. Slicing and filtering happen after lookup,
/// independently for each reader; no cache lock is held while decoding.
#[derive(Clone)]
pub(crate) struct DecodedSegments(Arc<DecodedCache>);

const DECODED_SHARDS: usize = 4;

struct DecodedCache {
    shards: [Mutex<DecodedShard>; DECODED_SHARDS],
    capacity: u64,
}

#[derive(Default)]
struct DecodedShard {
    arrays: FxHashMap<PlanByPtr, (ArrayRef, u64)>,
    order: VecDeque<PlanByPtr>,
    weight: u64,
}

impl DecodedSegments {
    fn shard(&self, plan: &PlanByPtr) -> &Mutex<DecodedShard> {
        let hash = FxBuildHasher.hash_one(plan);
        &self.0.shards[usize::from(hash.to_le_bytes()[0]) % DECODED_SHARDS]
    }

    pub(crate) fn get(&self, plan: PlanRef) -> Option<ArrayRef> {
        let key = PlanByPtr(plan);
        self.shard(&key)
            .lock()
            .arrays
            .get(&key)
            .map(|(array, _)| array.clone())
    }

    pub(crate) fn insert(&self, plan: PlanRef, array: ArrayRef) {
        // Include metadata even for arrays with no buffers, and weigh outside the shard lock.
        let weight = array.nbytes().saturating_add(1024);
        if weight > self.0.capacity {
            return;
        }
        let key = PlanByPtr(plan);
        let mut retired = Vec::new();
        {
            let mut shard = self.shard(&key).lock();
            if shard.arrays.contains_key(&key) {
                return;
            }
            // Readers progress through the file, so FIFO eviction needs no bookkeeping on hits.
            while shard.weight > self.0.capacity - weight {
                let Some(oldest) = shard.order.pop_front() else {
                    break;
                };
                if let Some((array, weight)) = shard.arrays.remove(&oldest) {
                    shard.weight -= weight;
                    retired.push((oldest, array));
                }
            }
            shard.weight += weight;
            shard.order.push_back(key.clone());
            shard.arrays.insert(key, (array, weight));
        }
        // Releasing an encoded array can walk its children; never do that under the shard lock.
        drop(retired);
    }
}

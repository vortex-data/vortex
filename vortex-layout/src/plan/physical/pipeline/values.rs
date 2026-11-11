// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Whole values, such as dictionary values, and decoded segments, kept between the scans that
//! share them.

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;

use crate::plan::physical::PlanByPtr;
use crate::plan::physical::PlanRef;
use crate::segments::SegmentId;

/// Values a plan produces over its whole domain whatever rows it is compiled for, kept for the
/// later scans that share this cache.
///
/// A dictionary's values are read whole by every reader of its codes, so readers sharing this
/// cache can reuse them across filters, projections, and splits. The caller controls its lifetime;
/// only whole values that a reader requested are retained.
///
/// It can also keep recently decoded segments, up to a budget in segment bytes, so a later scan
/// that reads a segment again takes its array as it is, with no read and no second validation
/// and decoding of its flatbuffer. The arrays share the segments' buffers, so the budget is about
/// the memory they hold. The default keeps none.
#[derive(Clone, Default)]
pub struct ValueCache {
    values: Arc<Mutex<FxHashMap<PlanByPtr, ArrayRef>>>,
    segments: Arc<Mutex<DecodedSegments>>,
}

impl ValueCache {
    /// A cache that also keeps up to `bytes` of decoded segments.
    pub fn with_segment_bytes(bytes: u64) -> Self {
        Self {
            values: Arc::default(),
            segments: Arc::new(Mutex::new(DecodedSegments {
                budget: bytes,
                ..DecodedSegments::default()
            })),
        }
    }

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

    /// The whole array of `segment`, if a scan decoded it recently.
    pub(crate) fn segment(&self, segment: SegmentId) -> Option<ArrayRef> {
        self.segments.lock().arrays.get(&segment).cloned()
    }

    /// Keeps the whole array decoded from `segment`, whose bytes are `bytes` long, evicting the
    /// oldest kept segments to stay within budget.
    pub(crate) fn insert_segment(&self, segment: SegmentId, array: &ArrayRef, bytes: u64) {
        let mut segments = self.segments.lock();
        if bytes > segments.budget || segments.arrays.contains_key(&segment) {
            return;
        }
        while segments.bytes + bytes > segments.budget {
            let Some((oldest, size)) = segments.order.pop_front() else {
                break;
            };
            segments.arrays.remove(&oldest);
            segments.bytes -= size;
        }
        segments.arrays.insert(segment, array.clone());
        segments.order.push_back((segment, bytes));
        segments.bytes += bytes;
    }
}

/// Decoded segments, oldest first in `order`.
#[derive(Default)]
struct DecodedSegments {
    budget: u64,
    arrays: FxHashMap<SegmentId, ArrayRef>,
    order: VecDeque<(SegmentId, u64)>,
    bytes: u64,
}

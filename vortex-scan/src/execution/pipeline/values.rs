// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Whole values, such as dictionary values, kept between the scans that share them.

use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::plan::PlanByPtr;
use crate::plan::PlanRef;
use crate::source::SegmentId;

/// Values a plan produces over its whole domain whatever rows it is compiled for, kept for the
/// later scans that share this cache.
///
/// A dictionary's values are read whole by every reader of its codes, so readers sharing this
/// cache can reuse them across filters, projections, and splits. The caller controls its lifetime;
/// only whole values that a reader requested are retained.
///
/// It also keeps each decoded segment that chunks read only part of, until they have read all of
/// its rows, so the chunks of a segment decode it once between them.
#[derive(Clone, Default)]
pub struct ValueCache(Arc<Inner>);

#[derive(Default)]
struct Inner {
    values: Mutex<FxHashMap<PlanByPtr, ArrayRef>>,
    segments: Mutex<FxHashMap<SegmentId, Decoded>>,
}

/// A decoded segment, and how many of its rows no chunk has read yet.
struct Decoded {
    array: ArrayRef,
    unread: usize,
}

impl ValueCache {
    /// The values `plan` produced earlier.
    pub(crate) fn get(&self, plan: &PlanRef) -> Option<ArrayRef> {
        self.0.values.lock().get(&PlanByPtr(plan.clone())).cloned()
    }

    /// Records the values of `plan`, keeping the first when two race.
    pub(crate) fn insert(&self, plan: &PlanRef, values: ArrayRef) -> ArrayRef {
        self.0
            .values
            .lock()
            .entry(PlanByPtr(plan.clone()))
            .or_insert(values)
            .clone()
    }

    /// The whole of `segment`, of `rows` rows, decoded by `decode` unless a chunk that read
    /// part of it did so before, for a chunk that reads `read` of its rows.
    ///
    /// The decoded segment is kept until chunks have read as many rows as it has, so a segment
    /// whose chunks are pruned is kept as long as the cache. Two chunks that miss at once both
    /// decode it.
    pub(crate) fn segment(
        &self,
        segment: SegmentId,
        rows: usize,
        read: usize,
        decode: impl FnOnce() -> VortexResult<ArrayRef>,
    ) -> VortexResult<ArrayRef> {
        if read >= rows {
            return decode();
        }
        if let Some(array) = self.take_rows(segment, read) {
            return Ok(array);
        }
        let array = decode()?;
        let mut segments = self.0.segments.lock();
        let decoded = segments.entry(segment).or_insert(Decoded {
            array,
            unread: rows,
        });
        decoded.unread = decoded.unread.saturating_sub(read);
        let array = decoded.array.clone();
        if decoded.unread == 0 {
            segments.remove(&segment);
        }
        Ok(array)
    }

    /// The decoded `segment`, if kept, counting `read` more of its rows read.
    fn take_rows(&self, segment: SegmentId, read: usize) -> Option<ArrayRef> {
        let mut segments = self.0.segments.lock();
        let decoded = segments.get_mut(&segment)?;
        decoded.unread = decoded.unread.saturating_sub(read);
        let array = decoded.array.clone();
        if decoded.unread == 0 {
            segments.remove(&segment);
        }
        Some(array)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::ValueCache;
    use crate::source::SegmentId;

    #[test]
    fn chunks_of_a_segment_decode_it_once() -> VortexResult<()> {
        let cache = ValueCache::default();
        let decodes = Cell::new(0);
        let decode = || {
            decodes.set(decodes.get() + 1);
            Ok(buffer![0u32, 1, 2, 3].into_array())
        };
        let segment = SegmentId::from(7u32);
        for _ in 0..4 {
            cache.segment(segment, 4, 1, decode)?;
        }
        assert_eq!(decodes.get(), 1);
        // Every row has been read, so the segment is no longer kept.
        assert!(cache.0.segments.lock().is_empty());
        cache.segment(segment, 4, 4, decode)?;
        assert_eq!(decodes.get(), 2);
        Ok(())
    }
}

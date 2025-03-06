use std::collections::BTreeMap;
use std::fmt::Display;
use std::ops::Deref;
use std::sync::{Arc, RwLock, RwLockWriteGuard};

use async_trait::async_trait;
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult};

use crate::range_intersection;

/// The identifier for a single segment.
// TODO(ngates): should this be a `[u8]` instead? Allowing for arbitrary segment identifiers?
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentId(u32);

impl From<u32> for SegmentId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl Deref for SegmentId {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for SegmentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SegmentId({})", self.0)
    }
}

#[async_trait]
pub trait AsyncSegmentReader: 'static + Send + Sync {
    /// Attempt to get the data associated with a given segment ID.
    async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer>;
}

pub trait SegmentWriter {
    /// Write the given data into a segment and return its identifier.
    /// The provided buffers are concatenated together to form the segment.
    ///
    // TODO(ngates): in order to support aligned Direct I/O, it is preferable for all segments to
    //  be aligned to the logical block size (typically 512, but could be 4096). For this reason,
    //  if we know we're going to read an entire FlatLayout together, then we should probably
    //  serialize it into a single segment that is 512 byte aligned? Or else, we should guarantee
    //  to align the the first segment to 512, and then assume that coalescing captures the rest.
    fn put(&mut self, buffer: &[ByteBuffer]) -> SegmentId;
}

#[derive(Debug, Default, PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord)]
pub enum RequiredSegmentKind {
    PRUNING = 1,
    FILTER = 2,
    #[default]
    PROJECTION = 3,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord)]
pub struct SegmentPriority {
    row_end: u64, // sort by row_end first
    kind: RequiredSegmentKind,
    row_start: u64,
}

type SegmentStore = BTreeMap<SegmentPriority, Vec<SegmentId>>;

#[derive(Default, Debug)]
pub struct SegmentRegistry {
    store: Arc<RwLock<SegmentStore>>,
    pub kind: RequiredSegmentKind,
}

impl SegmentRegistry {
    pub fn with_priority_hint(&self, kind: RequiredSegmentKind) -> Self {
        SegmentRegistry {
            store: self.store.clone(),
            // highest priority wins
            kind: kind.min(self.kind),
        }
    }

    pub fn push(&mut self, row_start: u64, row_end: u64, segment: SegmentId) {
        let (start, end) = match self.kind {
            // row offset inside the stats table is not our concern
            RequiredSegmentKind::PRUNING => (0, 0),
            _ => (row_start, row_end),
        };
        let priority = SegmentPriority {
            row_start: start,
            row_end: end,
            kind: self.kind,
        };
        self.write().entry(priority).or_default().push(segment);
    }

    pub fn retain_matching(&self, row_indices: Buffer<u64>) {
        if row_indices.is_empty() {
            return;
        }
        self.write().retain(|key, _segments| {
            if key.kind == RequiredSegmentKind::PRUNING {
                return true; // keep segments required for pruning
            }
            range_intersection(&(key.row_start..key.row_end), &row_indices).is_some()
        });
    }

    pub fn into_inner(self) -> BTreeMap<SegmentPriority, Vec<SegmentId>> {
        match Arc::try_unwrap(self.store) {
            Ok(store) => store.into_inner().vortex_expect("poisoned lock"),
            Err(arc_store) => arc_store.read().vortex_expect("poisoned lock").clone(),
        }
    }

    fn write(&self) -> RwLockWriteGuard<SegmentStore> {
        self.store.write().vortex_expect("poisoned lock")
    }
}

#[cfg(test)]
pub mod test {
    use vortex_buffer::ByteBufferMut;
    use vortex_error::{VortexExpect, vortex_err};

    use super::*;

    #[derive(Default)]
    pub struct TestSegments {
        segments: Vec<ByteBuffer>,
    }

    impl SegmentWriter for TestSegments {
        fn put(&mut self, data: &[ByteBuffer]) -> SegmentId {
            let id = u32::try_from(self.segments.len())
                .vortex_expect("Cannot store more than u32::MAX segments");

            // Combine all the buffers since we're only a test implementation
            let mut buffer = ByteBufferMut::empty();
            for segment in data {
                buffer.extend_from_slice(segment.as_ref());
            }
            self.segments.push(buffer.freeze());

            id.into()
        }
    }

    #[async_trait]
    impl AsyncSegmentReader for TestSegments {
        async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer> {
            self.segments
                .get(*id as usize)
                .cloned()
                .ok_or_else(|| vortex_err!("Segment not found"))
        }
    }
}

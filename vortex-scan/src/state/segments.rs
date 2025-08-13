// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use dashmap::DashMap;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult};
use vortex_layout::segments::{SegmentId, Segments};
use vortex_utils::aliases::hash_map::HashMap;

/// The working set of segments used by the scan.
#[derive(Default)]
pub(super) struct SegmentCache {
    ref_counts: HashMap<SegmentId, usize>,
    working_set: Arc<DashMap<SegmentId, ByteBuffer>>,
    working_set_size: u64,
}

impl SegmentCache {
    pub(super) fn segments(&self) -> Arc<dyn Segments> {
        self.working_set.clone()
    }

    pub(super) fn insert(&mut self, segment_id: SegmentId, buffer: ByteBuffer) {
        if self.ref_counts.contains_key(&segment_id) {
            self.working_set_size += buffer.len() as u64;
            self.working_set.insert(segment_id, buffer);
        }
    }

    pub(super) fn contains(&self, segment_id: &SegmentId) -> bool {
        self.working_set.contains_key(segment_id)
    }

    pub(super) fn len(&self) -> usize {
        self.working_set.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.working_set.is_empty()
    }

    pub(super) fn ref_counts(&self) -> &HashMap<SegmentId, usize> {
        &self.ref_counts
    }

    pub(super) fn acquire<'a, I: IntoIterator<Item = &'a SegmentId>>(&mut self, segment_ids: I) {
        for segment_id in segment_ids.into_iter() {
            *self.ref_counts.entry(*segment_id).or_default() += 1;
        }
    }

    /// Release the reference to the given segments, dropping any fetched buffers if possible.
    pub(super) fn release<'a, I: IntoIterator<Item = &'a SegmentId>>(&mut self, segment_ids: I) {
        for segment_id in segment_ids.into_iter() {
            let ref_count = self
                .ref_counts
                .get(segment_id)
                .vortex_expect("unknown segment");
            if *ref_count == 1 {
                if let Some((_, buffer)) = self.working_set.remove(segment_id) {
                    self.working_set_size -= buffer.len() as u64;
                }
                self.ref_counts.remove(segment_id);
            } else {
                let ref_count = self
                    .ref_counts
                    .get_mut(segment_id)
                    .vortex_expect("unknown segment");
                *ref_count -= 1;
            }
        }
    }
}

/// An opaque segment request returned from a `SegmentSource`.
pub trait SegmentRequest {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn size(&mut self) -> usize;
}

pub trait SegmentSource2 {
    /// Return the byte size of the given segment.
    fn size(&self, segment_id: &SegmentId) -> usize;

    /// Return an empty segment request.
    fn empty_request(&self) -> Box<dyn SegmentRequest>;

    /// Request to add the given segment into the request object.
    fn request_segment(&self, segment_id: &SegmentId, request: &mut dyn SegmentRequest) -> bool;

    /// Request the given segments.
    fn submit(&self, request: Box<dyn SegmentRequest>, callback: Arc<dyn SegmentCallback>);
}

pub trait SegmentCallback {
    fn on_segment(&self, segment_id: &SegmentId, buffer: VortexResult<ByteBuffer>);
}

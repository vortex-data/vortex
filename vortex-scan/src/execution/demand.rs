// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads each segment a scan's chunks read once, and keeps it until every one of them has.

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::buffer::BufferHandle;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::filter::segments_read;
use crate::plan::PlanRef;
use crate::source::SegmentFuture;
use crate::source::SegmentId;
use crate::source::SegmentSource;

type SharedRead = Shared<BoxFuture<'static, SharedVortexResult<BufferHandle>>>;

struct Demand {
    /// The segment's read, once requested.
    read: Option<SharedRead>,
    /// The chunks that have not yet taken the segment.
    remaining: u32,
}

/// Serves each segment a scan's chunks read from one read, shared by those chunks.
///
/// A scan reads through one IO session per chunk, so a segment longer than a chunk is requested
/// by every chunk it overlaps, often after the previous chunk's read has finished. This source
/// keeps the read until the last of those chunks has taken it. Segments are requested only when
/// a chunk first asks for them, so each chunk's IO session still coalesces and pipelines its own
/// reads. Segments of a chunk the scan skips stay counted until the scan is dropped.
///
/// Requests for segments no chunk was counted for go straight to `inner`.
pub(crate) struct DemandSegmentSource {
    inner: Arc<dyn SegmentSource>,
    demand: Mutex<FxHashMap<SegmentId, Demand>>,
}

impl DemandSegmentSource {
    /// `segments` lists each segment with the number of chunks that read it.
    pub(crate) fn new(inner: Arc<dyn SegmentSource>, segments: FxHashMap<SegmentId, u32>) -> Self {
        let demand = segments
            .into_iter()
            .map(|(id, chunks)| {
                let demand = Demand {
                    read: None,
                    remaining: chunks,
                };
                (id, demand)
            })
            .collect();
        Self {
            inner,
            demand: Mutex::new(demand),
        }
    }
}

impl SegmentSource for DemandSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let mut demand = self.demand.lock();
        let Some(entry) = demand.get_mut(&id) else {
            drop(demand);
            return self.inner.request(id);
        };
        let shared = entry
            .read
            .get_or_insert_with(|| read(&self.inner, id))
            .clone();
        entry.remaining -= 1;
        if entry.remaining == 0 {
            demand.remove(&id);
        }
        shared.map_err(VortexError::from).boxed()
    }
}

fn read(inner: &Arc<dyn SegmentSource>, id: SegmentId) -> SharedRead {
    inner.request(id).map_err(Arc::new).boxed().shared()
}

/// The segments `plans` read over each of `chunks`, with how many chunks read each. Compiling
/// finds them without IO.
pub(crate) fn chunk_segments<'a>(
    session: &VortexSession,
    plans: impl Iterator<Item = &'a PlanRef> + Clone,
    chunks: &[Range<u64>],
) -> VortexResult<FxHashMap<SegmentId, u32>> {
    let mut segments = FxHashMap::default();
    for rows in chunks {
        let len = usize::try_from(rows.end - rows.start)?;
        for id in segments_read(session, plans.clone(), rows.clone(), &Mask::new_true(len))? {
            *segments.entry(id).or_insert(0) += 1;
        }
    }
    Ok(segments)
}

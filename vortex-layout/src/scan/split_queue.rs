// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The queue of splits behind an opened layout scan's stream.

use std::iter;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::vec;

use futures::StreamExt;
use futures::future;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;
use vortex_scan::stream::Claims;

use crate::scan::repeated_scan::ScanTasks;

/// One split of a scan, which returns the split's array, or `None` when the filter removes all
/// of its rows.
type SplitTask = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

/// A queue of the split futures of [`RepeatedScan::execute`], in row order, which the clones of
/// an opened scan's stream share. Each clone polls one worker over the queue.
///
/// Each worker claims splits from the queue and keeps up to `per_worker` of its own in
/// flight, spawned on the session's runtime, yielding them as they finish. This is the scan's
/// stream pipeline with its in-flight bound split across workers: `per_worker` is the per-thread
/// count that the stream multiplies by the cores, so a host polling one worker per core keeps as
/// many splits in flight as the stream does.
///
/// A worker yields only the splits it claimed, and claims a split only when it has room for one,
/// so no worker waits on a split another one claimed.
///
/// A worker with room for one split has nothing to overlap it with, so it polls the split itself
/// rather than spawning it. The split then runs on the thread that polls the worker, which also
/// consumes its array, and no other thread driving the runtime picks it up.
///
/// [`RepeatedScan::execute`]: crate::scan::repeated_scan::RepeatedScan::execute
pub(crate) struct SplitQueue {
    handle: Handle,
    per_worker: usize,
    splits: Mutex<vec::IntoIter<SplitTask>>,
    total: usize,
    claimed: AtomicUsize,
}

impl SplitQueue {
    pub(crate) fn new(handle: Handle, per_worker: usize, splits: ScanTasks<ArrayRef>) -> Self {
        Self {
            handle,
            per_worker: per_worker.max(1),
            total: splits.len(),
            splits: Mutex::new(splits.into_iter()),
            claimed: AtomicUsize::new(0),
        }
    }

    /// Claims the next split, if any is left.
    fn claim(&self) -> Option<SplitTask> {
        let split = self.splits.lock().next()?;
        self.claimed.fetch_add(1, Ordering::Relaxed);
        Some(split)
    }

    /// How many of the splits the stream's clones have claimed, out of how many in all.
    pub(crate) fn claims(&self) -> Claims {
        Claims {
            claimed: self.claimed.load(Ordering::Relaxed).min(self.total),
            total: self.total,
        }
    }

    /// A new worker over the shared queue, which one clone of the scan's stream polls.
    pub(crate) fn worker(self: Arc<Self>) -> BoxStream<'static, VortexResult<ArrayRef>> {
        let handle = self.handle.clone();
        let per_worker = self.per_worker;
        let splits = stream::iter(iter::from_fn(move || self.claim()));
        if per_worker == 1 {
            return splits
                .then(|split| split)
                .filter_map(|array| future::ready(array.transpose()))
                .boxed();
        }
        // `buffer_unordered` pulls a split only when it has room, so a worker claims no more
        // than it runs.
        splits
            .map(move |split| handle.spawn(split))
            .buffer_unordered(per_worker)
            .filter_map(|array| future::ready(array.transpose()))
            .boxed()
    }
}

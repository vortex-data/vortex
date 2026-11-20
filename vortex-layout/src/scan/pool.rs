// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The pool an opened layout scan shares among host threads.

use std::iter;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::vec;

use futures::StreamExt;
use futures::future;
use futures::future::BoxFuture;
use futures::stream;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;
use vortex_scan::pool::ScanPool;

use crate::scan::repeated_scan::ScanTasks;

/// One split of a scan, which returns the split's array, or `None` when the filter removes all
/// of its rows.
type SplitTask = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

/// A pool over the split futures of [`RepeatedScan::execute`], in row order.
///
/// Each worker claims splits from the shared queue and keeps up to `per_worker` of its own in
/// flight, spawned on the session's runtime, yielding them as they finish. This is the scan's
/// stream pipeline with its in-flight bound split across workers: `per_worker` is the per-thread
/// count that the stream multiplies by the cores, so a host polling one worker per core keeps as
/// many splits in flight as the stream does.
///
/// A worker yields only the splits it claimed, and claims a split only when it has room for one.
/// So [`is_exhausted`](ScanPool::is_exhausted) turns true once every split has been claimed, and
/// a host can move a thread to other work then: no worker waits on a split another one claimed.
///
/// [`RepeatedScan::execute`]: crate::scan::repeated_scan::RepeatedScan::execute
pub(crate) struct SplitPool {
    dtype: DType,
    handle: Handle,
    per_worker: usize,
    splits: Mutex<vec::IntoIter<SplitTask>>,
    total: usize,
    claimed: AtomicUsize,
}

impl SplitPool {
    pub(crate) fn new(
        dtype: DType,
        handle: Handle,
        per_worker: usize,
        splits: ScanTasks<ArrayRef>,
    ) -> Self {
        Self {
            dtype,
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
        self.claimed.fetch_add(1, Ordering::Release);
        Some(split)
    }
}

impl ScanPool for SplitPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let dtype = self.dtype.clone();
        let handle = self.handle.clone();
        let per_worker = self.per_worker;
        // `buffer_unordered` pulls a split only when it has room, so a worker claims no more
        // than it runs.
        let arrays = stream::iter(iter::from_fn(move || self.claim()))
            .map(move |split| handle.spawn(split))
            .buffer_unordered(per_worker)
            .filter_map(|array| future::ready(array.transpose()));
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, arrays))
    }

    fn progress(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        (self.claimed.load(Ordering::Acquire) as f64 / self.total as f64).clamp(0.0, 1.0)
    }

    fn is_exhausted(&self) -> bool {
        self.claimed.load(Ordering::Acquire) >= self.total
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The pool an opened layout scan shares among host threads.

use std::iter;
use std::sync::Arc;
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
/// A worker yields only the splits it claimed, and claims a split only when it has room for one,
/// so no worker waits on a split another one claimed.
///
/// [`RepeatedScan::execute`]: crate::scan::repeated_scan::RepeatedScan::execute
pub(crate) struct SplitPool {
    dtype: DType,
    handle: Handle,
    per_worker: usize,
    splits: Mutex<vec::IntoIter<SplitTask>>,
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
            splits: Mutex::new(splits.into_iter()),
        }
    }

    /// Claims the next split, if any is left.
    fn claim(&self) -> Option<SplitTask> {
        self.splits.lock().next()
    }
}

impl ScanPool for SplitPool {
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
}

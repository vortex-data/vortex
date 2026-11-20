// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The pool an opened layout scan shares among host threads.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::vec;

use futures::future::BoxFuture;
use futures::stream;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_scan::pool::ScanPool;

use crate::scan::repeated_scan::ScanTasks;

/// One split of a scan, which returns the split's array, or `None` when the filter removes all
/// of its rows.
type SplitTask = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

/// A pool over the split futures of [`RepeatedScan::execute`], in row order.
///
/// A worker claims the next split and then runs it itself, outside the lock, one split at a
/// time. This is how DuckDB's threads read a file's splits on develop, so a host that runs one
/// worker per thread reads as many splits at once as it has threads, with no read-ahead.
///
/// [`RepeatedScan::execute`]: crate::scan::repeated_scan::RepeatedScan::execute
pub(crate) struct SplitPool {
    dtype: DType,
    splits: Mutex<vec::IntoIter<SplitTask>>,
    total: usize,
    claimed: AtomicUsize,
}

impl SplitPool {
    pub(crate) fn new(dtype: DType, splits: ScanTasks<ArrayRef>) -> Self {
        Self {
            dtype,
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
        let arrays = stream::unfold(self, |pool| async move {
            loop {
                match pool.claim()?.await {
                    Ok(Some(array)) => return Some((Ok(array), pool)),
                    Ok(None) => {}
                    Err(err) => return Some((Err(err), pool)),
                }
            }
        });
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

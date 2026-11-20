// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pools of work that host threads share by each polling a worker, and the pieces that sources
//! build their own pools from.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use async_lock::Mutex;
use futures::StreamExt;
use futures::future;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Precision;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::DataSourceScanRef;
use crate::PartitionStream;

/// A shared reference to a [`ScanPool`].
pub type ScanPoolRef = Arc<dyn ScanPool>;

/// The shared work of an opened scan, which host threads read by each polling a worker.
pub trait ScanPool: 'static + Send + Sync {
    /// The dtype of the arrays every worker returns.
    fn dtype(&self) -> &DType;

    /// Returns a new worker: a stream that one host thread polls.
    ///
    /// Workers take work from the pool until none is left, so together they return every row of
    /// the scan exactly once. A host may start any number of workers, at any time, and the number
    /// it runs is its parallelism. A pool may run its work as tasks on the session's runtime,
    /// which the host owns, and hand each result to whichever worker asks next; it never starts
    /// threads of its own. A host whose runtime has no threads of its own, such as a
    /// `CurrentThreadRuntime`, must therefore drive that runtime while it waits on a worker, for
    /// example with `block_on`. Dropping a worker before it ends gives up any work it had taken,
    /// so a host drops a worker early only when it abandons the scan.
    fn worker(self: Arc<Self>) -> SendableArrayStream;

    /// The fraction of the pool's work that workers have taken, from 0.0 to 1.0, for a host's
    /// progress bar. Never blocks.
    fn progress(&self) -> f64 {
        0.0
    }

    /// Whether workers have taken every unit of the pool, so a new worker would return nothing.
    ///
    /// A host that moves threads between pools uses it to stop starting workers on a drained
    /// pool. `false` is always a safe answer. Never blocks.
    fn is_exhausted(&self) -> bool {
        false
    }
}

/// A cloneable row budget shared by all work that can contribute rows to one scan.
///
/// Filterless scans can reserve each unit's selected rows before decoding it, so rows past the
/// limit are never decoded. Filtered scans cannot know a unit's row count up front, so they take
/// rows from the budget as arrays are emitted and discard the excess. Either way, the order in
/// which rows are taken decides which rows are returned: an ordered scan takes them in row order,
/// an unordered one in completion order.
#[derive(Clone, Debug)]
pub struct RowLimit(Arc<AtomicU64>);

impl RowLimit {
    /// A budget of `limit` rows.
    pub fn new(limit: u64) -> Self {
        Self(Arc::new(AtomicU64::new(limit)))
    }

    /// Reserves the rows selected by `mask` and keeps only the earliest granted rows of it.
    pub fn limit(&self, mask: Mask) -> Mask {
        let granted = self.take(mask.true_count());
        mask.limit(granted)
    }

    /// Reserves up to `rows` rows, returning how many the remaining budget granted.
    pub fn take(&self, rows: usize) -> usize {
        let requested = u64::try_from(rows).unwrap_or(u64::MAX);
        usize::try_from(self.reserve(requested)).unwrap_or(usize::MAX)
    }

    /// Whether the budget is spent.
    pub fn is_exhausted(&self) -> bool {
        self.0.load(Ordering::Relaxed) == 0
    }

    fn reserve(&self, requested: u64) -> u64 {
        let remaining = self
            .0
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                Some(remaining.saturating_sub(requested))
            })
            .ok()
            .vortex_expect("row reservation always updates the budget");
        remaining.min(requested)
    }
}

/// Ends `arrays` once `limit` is spent, taking each array's rows from it as the array is yielded
/// and slicing the array that spends it. Also ends after the first error.
pub fn trim_to_limit(
    arrays: BoxStream<'static, VortexResult<ArrayRef>>,
    limit: RowLimit,
) -> BoxStream<'static, VortexResult<ArrayRef>> {
    stream::unfold(Some(arrays), move |arrays| {
        let limit = limit.clone();
        async move {
            let mut arrays = arrays?;
            if limit.is_exhausted() {
                return None;
            }
            let array = match arrays.next().await? {
                Ok(array) => array,
                Err(err) => return Some((Err(err), None)),
            };
            let granted = limit.take(array.len());
            if granted == array.len() {
                return Some((Ok(array), Some(arrays)));
            }
            if granted == 0 {
                return None;
            }
            Some((array.slice(0..granted), None))
        }
    })
    .filter(|array| future::ready(array.as_ref().map_or(true, |array| !array.is_empty())))
    .boxed()
}

/// Groups consecutive `items` into at most `n` contiguous groups of about equal total `weight`.
///
/// Never cuts an item: each lands whole in one group. Every group is non-empty.
pub fn group_contiguous<T>(
    items: Vec<T>,
    n: NonZeroUsize,
    weight: impl Fn(&T) -> u64,
) -> Vec<Vec<T>> {
    let total = items
        .iter()
        .map(|item| u128::from(weight(item)))
        .sum::<u128>()
        .max(1);
    let n_groups = n.get() as u128;
    let mut groups: Vec<Vec<T>> = Vec::new();
    let mut current_group = None;
    let mut before = 0u128;
    for item in items {
        let group = (before * n_groups / total).min(n_groups - 1);
        before += u128::from(weight(&item));
        if current_group != Some(group) {
            current_group = Some(group);
            groups.push(Vec::new());
        }
        if let Some(last) = groups.last_mut() {
            last.push(item);
        }
    }
    groups
}

/// A pool whose workers take partitions one at a time from a scan's shared partition stream.
///
/// This is the default [`open`](crate::DataSourceScan::open) of every scan.
pub struct PartitionPool {
    dtype: DType,
    partitions: Mutex<PartitionStream>,
    /// The scan's partition count, if it is known or estimated.
    partition_count: Option<usize>,
    taken: AtomicUsize,
    exhausted: AtomicBool,
}

impl PartitionPool {
    /// Opens a pool over the partitions of `scan`.
    pub fn new(scan: DataSourceScanRef) -> Self {
        let partition_count = match scan.partition_count() {
            Precision::Exact(count) | Precision::Inexact(count) => Some(count),
            Precision::Absent => None,
        };
        Self {
            dtype: scan.dtype().clone(),
            partitions: Mutex::new(scan.partitions()),
            partition_count,
            taken: AtomicUsize::new(0),
            exhausted: AtomicBool::new(false),
        }
    }
}

impl ScanPool for PartitionPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let dtype = self.dtype.clone();
        let partitions = stream::unfold(self, |pool| async move {
            let Some(next) = pool.partitions.lock().await.next().await else {
                pool.exhausted.store(true, Ordering::Release);
                return None;
            };
            pool.taken.fetch_add(1, Ordering::Relaxed);
            Some((next, pool))
        });
        let arrays = partitions.flat_map(|partition| {
            match partition.and_then(|partition| partition.execute()) {
                Ok(arrays) => arrays.left_stream(),
                Err(err) => stream::once(async move { Err(err) }).right_stream(),
            }
        });
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, arrays))
    }

    fn progress(&self) -> f64 {
        if self.exhausted.load(Ordering::Acquire) {
            return 1.0;
        }
        match self.partition_count {
            Some(count) if count > 0 => {
                (self.taken.load(Ordering::Relaxed) as f64 / count as f64).clamp(0.0, 1.0)
            }
            _ => 0.0,
        }
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;
    use std::thread;

    use futures::StreamExt;
    use futures::TryStreamExt;
    use futures::executor::block_on;
    use futures::future::try_join_all;
    use futures::stream;
    use vortex_array::IntoArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::stats::Precision;
    use vortex_array::stream::ArrayStreamAdapter;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::stream::SendableArrayStream;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::RowLimit;
    use super::group_contiguous;
    use super::trim_to_limit;
    use crate::DataSourceScan;
    use crate::Partition;
    use crate::PartitionRef;
    use crate::PartitionStream;

    static DTYPE: DType = DType::Primitive(PType::I32, Nullability::NonNullable);

    /// A partition of `rows` rows, returned as one array.
    struct RowsPartition {
        index: usize,
        rows: usize,
    }

    impl Partition for RowsPartition {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn index(&self) -> usize {
            self.index
        }

        fn row_count(&self) -> Precision<u64> {
            Precision::exact(self.rows as u64)
        }

        fn byte_size(&self) -> Precision<u64> {
            Precision::Absent
        }

        fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
            let array = Buffer::from_iter(0..i32::try_from(self.rows)?).into_array();
            Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
                DTYPE.clone(),
                stream::iter([Ok(array)]),
            )))
        }
    }

    /// A scan that implements only the partition methods.
    struct PartitionOnlyScan {
        rows: Vec<usize>,
    }

    impl DataSourceScan for PartitionOnlyScan {
        fn dtype(&self) -> &DType {
            &DTYPE
        }

        fn partition_count(&self) -> Precision<usize> {
            Precision::exact(self.rows.len())
        }

        fn partitions(self: Box<Self>) -> PartitionStream {
            stream::iter(
                self.rows.into_iter().enumerate().map(|(index, rows)| {
                    Ok(Box::new(RowsPartition { index, rows }) as PartitionRef)
                }),
            )
            .boxed()
        }
    }

    #[test]
    fn default_open_shares_partitions_across_workers() -> VortexResult<()> {
        let scan = Box::new(PartitionOnlyScan {
            rows: vec![2, 3, 4, 5],
        });
        let pool = scan.open()?;
        let workers = (0..3).map(|_| {
            Arc::clone(&pool)
                .worker()
                .try_fold(0, |rows, array| async move { Ok(rows + array.len()) })
        });
        let rows: usize = block_on(try_join_all(workers))?.into_iter().sum();
        assert_eq!(rows, 14);
        Ok(())
    }

    /// Progress counts partitions as workers take them, and the pool is exhausted once a worker
    /// finds none left.
    #[test]
    fn default_pool_reports_progress_and_exhaustion() -> VortexResult<()> {
        let scan = Box::new(PartitionOnlyScan {
            rows: vec![2, 3, 4, 5],
        });
        let pool = scan.open()?;
        assert_eq!(pool.progress(), 0.0);
        assert!(!pool.is_exhausted());

        let mut worker = Arc::clone(&pool).worker();
        let first = block_on(worker.next()).transpose()?;
        assert_eq!(first.map(|array| array.len()), Some(2));
        assert_eq!(pool.progress(), 0.25);
        assert!(!pool.is_exhausted());

        let rest =
            block_on(worker.try_fold(0, |rows, array| async move { Ok(rows + array.len()) }))?;
        assert_eq!(rest, 12);
        assert_eq!(pool.progress(), 1.0);
        assert!(pool.is_exhausted());
        assert!(block_on(Arc::clone(&pool).worker().next()).is_none());
        Ok(())
    }

    #[test]
    fn reserve_grants_up_to_the_remaining_budget() {
        let limit = RowLimit::new(5);
        assert_eq!(limit.reserve(3), 3);
        assert!(!limit.is_exhausted());
        assert_eq!(limit.reserve(10), 2);
        assert!(limit.is_exhausted());
        assert_eq!(limit.reserve(1), 0);
    }

    #[test]
    fn limit_keeps_the_earliest_granted_rows() {
        let limit = RowLimit::new(2);
        let mask = Mask::from_iter([true, false, true, true, false, true]);
        let limited = limit.limit(mask);
        assert_eq!(limited.true_count(), 2);
        assert!(limited.value(0));
        assert!(limited.value(2));
        assert!(!limited.value(3));
        assert!(!limited.value(5));
        assert!(limit.is_exhausted());
    }

    #[test]
    fn concurrent_reservations_never_exceed_the_budget() {
        const THREADS: usize = 8;
        const PER_THREAD: u64 = 10_000;
        const LIMIT: u64 = 25_000;

        let limit = RowLimit::new(LIMIT);
        let granted_total = Arc::new(AtomicU64::new(0));
        let barrier = Arc::new(Barrier::new(THREADS));
        thread::scope(|scope| {
            for _ in 0..THREADS {
                let limit = limit.clone();
                let granted_total = Arc::clone(&granted_total);
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    let mut local = 0;
                    for _ in 0..PER_THREAD {
                        local += limit.reserve(1);
                    }
                    granted_total.fetch_add(local, Ordering::Relaxed);
                });
            }
        });
        assert_eq!(granted_total.load(Ordering::Relaxed), LIMIT);
        assert!(limit.is_exhausted());
    }

    #[test]
    fn trim_to_limit_slices_the_array_that_spends_it() -> VortexResult<()> {
        let arrays = stream::iter((0..4).map(|_| Ok(Buffer::from_iter(0..3i32).into_array())));
        let trimmed: Vec<_> =
            block_on(trim_to_limit(arrays.boxed(), RowLimit::new(7)).try_collect())?;
        assert_eq!(
            trimmed.iter().map(|a| a.len()).collect::<Vec<_>>(),
            vec![3, 3, 1]
        );
        Ok(())
    }

    #[test]
    fn group_contiguous_balances_weights() {
        let groups = group_contiguous(
            vec![4u64, 4, 4, 4, 1, 1],
            NonZeroUsize::new(3).unwrap_or(NonZeroUsize::MIN),
            |w| *w,
        );
        assert_eq!(groups, vec![vec![4, 4], vec![4], vec![4, 1, 1]]);
    }

    #[test]
    fn default_split_returns_the_scan() -> VortexResult<()> {
        let scan = Box::new(PartitionOnlyScan { rows: vec![2, 3] });
        let pieces = scan.split(NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN))?;
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].partition_count(), Precision::exact(2usize));
        Ok(())
    }
}

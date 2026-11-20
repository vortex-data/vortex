// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pools of work that host threads share by each polling a worker.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use async_lock::Mutex;
use futures::StreamExt;
use futures::stream;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Precision;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;

use crate::DataSourceScanRef;
use crate::Partition;
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

/// A pool whose workers take partitions one at a time from a scan's shared partition stream.
///
/// A source without a pool of its own can open into this one, as long as it implements
/// [`partitions`](crate::DataSourceScan::partitions) itself.
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

/// A piece of a split scan as a partition, which opens the piece and drains one worker of it.
///
/// This is how the default [`partitions`](crate::DataSourceScan::partitions) reads a scan.
pub(crate) struct ScanPartition {
    index: usize,
    scan: DataSourceScanRef,
}

impl ScanPartition {
    pub(crate) fn new(index: usize, scan: DataSourceScanRef) -> Self {
        Self { index, scan }
    }
}

impl Partition for ScanPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.index
    }

    fn row_count(&self) -> Precision<u64> {
        self.scan.row_count()
    }

    fn byte_size(&self) -> Precision<u64> {
        self.scan.byte_size()
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        Ok(self.scan.open()?.worker())
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::collections::VecDeque;
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use futures::StreamExt;
    use futures::TryStreamExt;
    use futures::executor::block_on;
    use futures::future::try_join_all;
    use futures::stream;
    use parking_lot::Mutex;
    use vortex_array::ArrayRef;
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

    use super::PartitionPool;
    use super::ScanPool;
    use super::ScanPoolRef;
    use crate::DataSourceScan;
    use crate::DataSourceScanRef;
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
            let array = rows_array(self.rows)?;
            Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
                DTYPE.clone(),
                stream::iter([Ok(array)]),
            )))
        }
    }

    /// The array of `rows` rows that a test partition or pool returns.
    fn rows_array(rows: usize) -> VortexResult<ArrayRef> {
        Ok(Buffer::from_iter(0..i32::try_from(rows)?).into_array())
    }

    /// A scan that has no pool of its own, so it opens into a [`PartitionPool`].
    struct PartitionOnlyScan {
        rows: Vec<usize>,
    }

    impl DataSourceScan for PartitionOnlyScan {
        fn dtype(&self) -> &DType {
            &DTYPE
        }

        fn split(self: Box<Self>, _n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
            Ok(vec![self])
        }

        fn open(self: Box<Self>) -> VortexResult<ScanPoolRef> {
            Ok(Arc::new(PartitionPool::new(self)))
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
    fn partition_pool_shares_partitions_across_workers() -> VortexResult<()> {
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
    fn partition_pool_reports_progress_and_exhaustion() -> VortexResult<()> {
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

    /// A scan that implements only `split` and `open`, with a pool of its own.
    struct SplitOnlyScan {
        rows: Vec<usize>,
    }

    impl DataSourceScan for SplitOnlyScan {
        fn dtype(&self) -> &DType {
            &DTYPE
        }

        fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
            let size = self.rows.len().div_ceil(n.get()).max(1);
            Ok(self
                .rows
                .chunks(size)
                .map(|rows| {
                    Box::new(SplitOnlyScan {
                        rows: rows.to_vec(),
                    }) as DataSourceScanRef
                })
                .collect())
        }

        fn open(self: Box<Self>) -> VortexResult<ScanPoolRef> {
            Ok(Arc::new(RowsPool {
                rows: Mutex::new(self.rows.into()),
            }))
        }
    }

    /// A pool whose workers each take the next count of rows.
    struct RowsPool {
        rows: Mutex<VecDeque<usize>>,
    }

    impl ScanPool for RowsPool {
        fn dtype(&self) -> &DType {
            &DTYPE
        }

        fn worker(self: Arc<Self>) -> SendableArrayStream {
            let arrays = stream::unfold(self, |pool| async move {
                let rows = pool.rows.lock().pop_front()?;
                Some((rows_array(rows), pool))
            });
            ArrayStreamExt::boxed(ArrayStreamAdapter::new(DTYPE.clone(), arrays))
        }
    }

    /// The default `partitions` reads a scan that implements only `split` and `open`, one
    /// partition per piece.
    #[test]
    fn default_partitions_open_each_piece() -> VortexResult<()> {
        let scan = Box::new(SplitOnlyScan {
            rows: vec![2, 3, 4, 5],
        });
        let partitions: Vec<PartitionRef> = block_on(scan.partitions().try_collect())?;
        assert!(!partitions.is_empty());
        for (index, partition) in partitions.iter().enumerate() {
            assert_eq!(partition.index(), index);
        }
        let rows = block_on(try_join_all(partitions.into_iter().map(
            |partition| async move {
                partition
                    .execute()?
                    .try_fold(0, |rows, array| async move { Ok(rows + array.len()) })
                    .await
            },
        )))?;
        assert_eq!(rows.into_iter().sum::<usize>(), 14);
        Ok(())
    }
}

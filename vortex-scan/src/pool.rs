// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pools of work that host threads share by each polling a worker.

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

use crate::DataSourceScanRef;
use crate::PartitionStream;

/// A shared reference to a [`ScanPool`].
pub type ScanPoolRef = Arc<dyn ScanPool>;

/// The shared work of an opened scan, which host threads read by each polling a worker.
pub trait ScanPool: 'static + Send + Sync {
    /// The dtype of the arrays every worker returns.
    fn dtype(&self) -> &DType;

    /// Returns a new worker: a stream its caller polls inline, on its own thread.
    ///
    /// Workers take work from the pool until none is left, so together they return every row of
    /// the scan exactly once. A host may start any number of workers, at any time. A worker may
    /// spawn its work onto the session's runtime, so its splits decode concurrently and their I/O
    /// starts early, but it starts no threads of its own. Dropping a worker before it ends gives
    /// up the work it had taken, so a host drops a worker early only when it abandons the scan.
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
/// This is the default [`open`](crate::DataSourceScan::open) of every scan.
pub struct PartitionPool {
    dtype: DType,
    partitions: Mutex<PartitionStream>,
    /// The scan's partition count, if it has an upper bound.
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
            let array = Buffer::from_iter(0..self.rows as i32).into_array();
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

    #[test]
    fn default_split_returns_the_scan() -> VortexResult<()> {
        let scan = Box::new(PartitionOnlyScan { rows: vec![2, 3] });
        let pieces = scan.split(NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN))?;
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].partition_count(), Precision::exact(2usize));
        Ok(())
    }
}

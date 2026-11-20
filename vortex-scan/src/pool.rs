// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pools of work that host threads share by each polling a worker.

use std::sync::Arc;

use async_lock::Mutex;
use futures::StreamExt;
use futures::stream;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;

use crate::DataSourceScanRef;
use crate::PartitionStream;

/// A shared reference to a [`ScanPool`].
pub type ScanPoolRef = Arc<dyn ScanPool>;

/// The shared work of an opened scan, which host threads read by each polling a worker.
pub trait ScanPool: 'static + Send + Sync {
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
}

/// A pool whose workers take partitions one at a time from a scan's shared partition stream.
///
/// A source without a pool of its own can open into this one.
pub struct PartitionPool {
    dtype: DType,
    partitions: Mutex<PartitionStream>,
}

impl PartitionPool {
    /// Opens a pool over the partitions of `scan`.
    pub fn new(scan: DataSourceScanRef) -> Self {
        Self {
            dtype: scan.dtype().clone(),
            partitions: Mutex::new(scan.partitions()),
        }
    }
}

impl ScanPool for PartitionPool {
    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let dtype = self.dtype.clone();
        let partitions = stream::unfold(self, |pool| async move {
            let next = pool.partitions.lock().await.next().await?;
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
}

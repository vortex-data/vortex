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
    /// the scan exactly once. A host may start any number of workers, at any time. A worker never
    /// spawns tasks of its own. Dropping a worker before it ends gives up the work it had taken,
    /// so a host drops a worker early only when it abandons the scan.
    fn worker(self: Arc<Self>) -> SendableArrayStream;
}

/// A pool whose workers take partitions one at a time from a shared partition stream.
pub(crate) struct PartitionPool {
    dtype: DType,
    partitions: Mutex<PartitionStream>,
}

impl PartitionPool {
    pub(crate) fn new(dtype: DType, partitions: PartitionStream) -> Self {
        Self {
            dtype,
            partitions: Mutex::new(partitions),
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

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`ScanPool`] over the partitions of a [`DataSourceScan`].

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use async_lock::Mutex;
use futures::StreamExt;
use futures::stream;
use futures::stream::Fuse;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;

use crate::DataSourceScanRef;
use crate::PartitionRef;
use crate::PartitionStream;
use crate::ScanPool;

/// A [`ScanPool`] whose units are a scan's partitions.
///
/// Workers share the scan's partition stream: each takes the next partition and polls its
/// [`execute`](crate::Partition::execute) stream inline until it ends. The pool itself never
/// spawns, though a partition's stream may.
///
/// The pool shares no row limit between workers beyond what the partitions themselves share, so
/// an ordered scan with a limit may return more rows than the limit, as its partitions do.
pub struct PartitionPool {
    dtype: DType,
    partitions: Mutex<Fuse<PartitionStream>>,
    partition_count: Option<usize>,
    handed_out: AtomicUsize,
    is_exhausted: AtomicBool,
}

impl PartitionPool {
    /// Opens a pool over the partitions of `scan`.
    pub fn new(scan: DataSourceScanRef) -> Self {
        let dtype = scan.dtype().clone();
        let partition_count = scan.partition_count().into_inner();
        Self {
            dtype,
            partitions: Mutex::new(scan.partitions().fuse()),
            partition_count,
            handed_out: AtomicUsize::new(0),
            is_exhausted: AtomicBool::new(false),
        }
    }

    /// Takes the next partition, waiting for the stream if another worker is polling it.
    async fn next_partition(&self) -> Option<VortexResult<PartitionRef>> {
        let mut partitions = self.partitions.lock().await;
        let partition = partitions.next().await;
        match &partition {
            Some(_) => {
                self.handed_out.fetch_add(1, Ordering::Relaxed);
            }
            None => self.is_exhausted.store(true, Ordering::Relaxed),
        }
        partition
    }
}

impl ScanPool for PartitionPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let dtype = self.dtype.clone();
        let arrays = stream::unfold(
            (self, None::<SendableArrayStream>),
            |(pool, mut current)| async move {
                loop {
                    if let Some(mut arrays) = current.take()
                        && let Some(item) = arrays.next().await
                    {
                        return Some((item, (pool, Some(arrays))));
                    }
                    let partition = match pool.next_partition().await? {
                        Ok(partition) => partition,
                        Err(err) => return Some((Err(err), (pool, None))),
                    };
                    match partition.execute() {
                        Ok(arrays) => current = Some(arrays),
                        Err(err) => return Some((Err(err), (pool, None))),
                    }
                }
            },
        );
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, arrays))
    }

    fn is_exhausted(&self) -> bool {
        self.is_exhausted.load(Ordering::Relaxed)
    }

    fn progress(&self) -> f64 {
        if self.is_exhausted() {
            return 1.0;
        }
        match self.partition_count {
            Some(count) if count > 0 => {
                (self.handed_out.load(Ordering::Relaxed) as f64 / count as f64).clamp(0.0, 1.0)
            }
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use futures::StreamExt;
    use futures::TryStreamExt;
    use futures::stream;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::stats::Precision;
    use vortex_array::scalar::Scalar;
    use vortex_array::stream::ArrayStreamAdapter;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::stream::SendableArrayStream;
    use vortex_error::VortexResult;

    use crate::DataSourceScan;
    use crate::Partition;
    use crate::PartitionRef;
    use crate::PartitionStream;

    fn dtype() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    /// A scan of `partitions` partitions that only implements the partitions API. Partition `i`
    /// returns one array of `i + 1` rows.
    struct PartitionOnlyScan {
        dtype: DType,
        partitions: usize,
    }

    struct ToyPartition {
        index: usize,
        dtype: DType,
    }

    impl Partition for ToyPartition {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn index(&self) -> usize {
            self.index
        }

        fn row_count(&self) -> Precision<u64> {
            Precision::exact(self.index as u64 + 1)
        }

        fn byte_size(&self) -> Precision<u64> {
            Precision::Absent
        }

        fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
            let array = ConstantArray::new(Scalar::from(1i32), self.index + 1).into_array();
            Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
                self.dtype,
                stream::iter([Ok(array)]),
            )))
        }
    }

    impl DataSourceScan for PartitionOnlyScan {
        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn partition_count(&self) -> Precision<usize> {
            Precision::exact(self.partitions)
        }

        fn partitions(self: Box<Self>) -> PartitionStream {
            let dtype = self.dtype;
            // Each partition arrives after the stream has once returned `Pending`, so workers
            // must wait for one another's polls.
            stream::iter(0..self.partitions)
                .then(move |index| {
                    let dtype = dtype.clone();
                    async move {
                        tokio::task::yield_now().await;
                        Ok(Box::new(ToyPartition { index, dtype }) as PartitionRef)
                    }
                })
                .boxed()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 3)]
    async fn default_open_shares_partitions_between_workers() -> VortexResult<()> {
        let scan = Box::new(PartitionOnlyScan {
            dtype: dtype(),
            partitions: 6,
        });
        assert!(scan.ordered());
        let pool = scan.open()?;
        let workers = (0..3)
            .map(|_| {
                let worker = Arc::clone(&pool).worker();
                tokio::spawn(async move {
                    worker
                        .map_ok(|array| array.len())
                        .try_collect::<Vec<_>>()
                        .await
                })
            })
            .collect::<Vec<_>>();
        let mut lengths = Vec::new();
        for worker in workers {
            lengths.extend(worker.await.expect("worker panicked")?);
        }
        lengths.sort_unstable();
        assert_eq!(lengths, [1, 2, 3, 4, 5, 6]);
        assert!(pool.is_exhausted());
        assert_eq!(pool.progress(), 1.0);
        Ok(())
    }

    #[test]
    fn default_split_returns_the_scan_whole() -> VortexResult<()> {
        let scan = Box::new(PartitionOnlyScan {
            dtype: dtype(),
            partitions: 6,
        });
        let pieces = scan.split(NonZeroUsize::new(4).expect("non-zero"))?;
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].partition_count(), Precision::exact(6usize));
        Ok(())
    }
}

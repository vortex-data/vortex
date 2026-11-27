// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An opened scan: the work its host threads share, and the streams that read it.

use std::fmt;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use async_lock::Mutex;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stats::Precision;
use vortex_array::stream::ArrayStream;
use vortex_error::VortexResult;

use crate::DataSourceScanRef;

/// Starts a new worker over the shared work of an opened scan.
type NewWorker = dyn Fn() -> BoxStream<'static, VortexResult<ArrayRef>> + Send + Sync;

/// A flag that an opened scan sets once no stream created afterwards would find work.
///
/// It is set at most once and never cleared, so hosts read it from a cache line that is almost
/// never written. Clones share the flag: a source keeps one clone where it hands out work and
/// gives another to its [`ScanExec`].
#[derive(Clone, Debug, Default)]
pub struct ExhaustedFlag(Arc<AtomicBool>);

impl ExhaustedFlag {
    /// A flag that is not yet set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks the scan exhausted. Later calls do nothing.
    pub fn set(&self) {
        if !self.0.load(Ordering::Relaxed) {
            self.0.store(true, Ordering::Release);
        }
    }

    /// Whether the flag is set.
    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// An opened scan: the work that its host threads share.
///
/// A host that runs its own threads, or whose partitions share the scan's work, opens the scan
/// once and has each thread or partition read a [`stream`](Self::stream) of it. The streams take
/// work until none is left, so together they return every row of the scan exactly once, in no
/// particular order across streams. The number of streams a host reads is its parallelism.
/// Clones of a `ScanExec` share the same work, so a host keeps one per scan wherever its threads
/// can reach it.
///
/// A host decides whether another thread should join the scan with
/// [`is_exhausted`](Self::is_exhausted), which never blocks and writes nothing, so it is cheap
/// under a host's own lock. It creates the joining thread's stream after releasing that lock.
#[derive(Clone)]
pub struct ScanExec {
    dtype: DType,
    exhausted: ExhaustedFlag,
    new_worker: Arc<NewWorker>,
}

impl ScanExec {
    /// An opened scan whose streams each read a worker from `new_worker`.
    ///
    /// The workers that `new_worker` returns must share one queue of work, so that together they
    /// return every row exactly once. Creating a worker must be cheap and read nothing: a worker
    /// takes its first work when it is first polled. The source sets `exhausted` once a worker
    /// created afterwards would find no work.
    pub fn new(
        dtype: DType,
        exhausted: ExhaustedFlag,
        new_worker: impl Fn() -> BoxStream<'static, VortexResult<ArrayRef>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            dtype,
            exhausted,
            new_worker: Arc::new(new_worker),
        }
    }

    /// An opened scan whose streams take the partitions of `scan` one at a time, each stream
    /// reading up to `read_ahead` of its partitions at once.
    ///
    /// A source that has no queue of work of its own can open into this one. A scan that reports
    /// exactly zero partitions opens exhausted. Otherwise the source does not know its work up
    /// front, so the scan is exhausted only once a stream finds the partitions have run out. A
    /// stream returns the arrays of the partitions it reads at once in no particular order, so an
    /// ordered scan reads ahead one partition.
    pub fn from_partitions(scan: DataSourceScanRef, read_ahead: NonZeroUsize) -> Self {
        let dtype = scan.dtype().clone();
        let exhausted = ExhaustedFlag::new();
        if matches!(scan.partition_count(), Precision::Exact(0)) {
            exhausted.set();
        }
        let partitions = Arc::new(Mutex::new(scan.partitions()));
        let flag = exhausted.clone();
        Self::new(dtype, exhausted, move || {
            let flag = flag.clone();
            stream::unfold(Arc::clone(&partitions), move |shared| {
                let flag = flag.clone();
                async move {
                    let Some(next) = shared.lock().await.next().await else {
                        flag.set();
                        return None;
                    };
                    Some((next, shared))
                }
            })
            .map(|partition| partition.and_then(|partition| partition.execute()))
            .try_flatten_unordered(Some(read_ahead.get()))
            .boxed()
        })
    }

    /// The dtype of the scan's arrays.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Whether a stream created now would find no work. Never blocks, and writes nothing.
    ///
    /// Once true it stays true. While it is false a new stream may still find nothing, because
    /// streams take their work only when first polled.
    pub fn is_exhausted(&self) -> bool {
        self.exhausted.is_set()
    }

    /// A new stream over the scan's remaining work, which one host thread or partition reads.
    pub fn stream(&self) -> ScanStream {
        ScanStream {
            dtype: self.dtype.clone(),
            worker: (self.new_worker)(),
        }
    }
}

impl fmt::Debug for ScanExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanExec")
            .field("dtype", &self.dtype)
            .field("exhausted", &self.is_exhausted())
            .finish_non_exhaustive()
    }
}

/// One worker over the work of a [`ScanExec`], which one host thread or partition reads.
///
/// The work may run as tasks on the session's runtime, which the host owns, and nothing starts
/// threads of its own. A host whose runtime has no threads of its own, such as a
/// `CurrentThreadRuntime`, must therefore drive that runtime while it waits on a stream, for
/// example with `block_on`. Dropping a stream before it ends gives up any work it had taken, so a
/// host drops a stream early only when it abandons the scan.
pub struct ScanStream {
    dtype: DType,
    worker: BoxStream<'static, VortexResult<ArrayRef>>,
}

impl fmt::Debug for ScanStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanStream")
            .field("dtype", &self.dtype)
            .finish_non_exhaustive()
    }
}

impl Stream for ScanStream {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.worker.poll_next_unpin(cx)
    }
}

impl ArrayStream for ScanStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::num::NonZeroUsize;

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
    use vortex_array::stats::Precision;
    use vortex_array::stream::ArrayStreamAdapter;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::stream::SendableArrayStream;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::ScanExec;
    use super::ScanStream;
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

    /// The array of `rows` rows that a test partition returns.
    fn rows_array(rows: usize) -> VortexResult<ArrayRef> {
        Ok(Buffer::from_iter(0..i32::try_from(rows)?).into_array())
    }

    /// A scan that has no queue of work of its own, so it opens into [`ScanExec::from_partitions`].
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

        fn open(self: Box<Self>) -> VortexResult<ScanExec> {
            Ok(ScanExec::from_partitions(self, NonZeroUsize::MIN))
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

    /// The rows that `streams` return together, each polled concurrently.
    fn total_rows(streams: impl IntoIterator<Item = ScanStream>) -> VortexResult<usize> {
        let workers = streams
            .into_iter()
            .map(|stream| stream.try_fold(0, |rows, array| async move { Ok(rows + array.len()) }));
        Ok(block_on(try_join_all(workers))?.into_iter().sum())
    }

    #[test]
    fn streams_share_partitions() -> VortexResult<()> {
        let exec = Box::new(PartitionOnlyScan {
            rows: vec![2, 3, 4, 5],
        })
        .open()?;
        assert!(!exec.is_exhausted());
        assert_eq!(
            total_rows([exec.stream(), exec.stream(), exec.stream()])?,
            14
        );
        assert!(exec.is_exhausted());
        assert_eq!(total_rows([exec.stream()])?, 0);
        Ok(())
    }

    /// A host skips a scan with no work as soon as it opens it, before any stream runs.
    #[test]
    fn a_scan_with_no_partitions_opens_exhausted() -> VortexResult<()> {
        let exec = Box::new(PartitionOnlyScan { rows: vec![] }).open()?;
        assert!(exec.is_exhausted());
        assert_eq!(total_rows([exec.stream()])?, 0);
        Ok(())
    }

    /// Streams that read ahead still return every row once.
    #[test]
    fn streams_reading_ahead_share_partitions() -> VortexResult<()> {
        let scan = Box::new(PartitionOnlyScan {
            rows: vec![2, 3, 4, 5],
        });
        let exec = ScanExec::from_partitions(scan, NonZeroUsize::MIN.saturating_add(1));
        assert_eq!(total_rows([exec.stream(), exec.stream()])?, 14);
        Ok(())
    }
}

//! The stream that an opened scan returns, which host threads share by each polling a clone.

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use async_lock::Mutex;
use futures::Stream;
use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStream;
use vortex_error::VortexResult;

use crate::DataSourceScanRef;

/// Starts a new worker over the shared work of an opened scan.
type NewWorker = dyn Fn() -> BoxStream<'static, VortexResult<ArrayRef>> + Send + Sync;

/// Reads how many units of an opened scan's work its clones have claimed.
type ReadClaims = dyn Fn() -> Claims + Send + Sync;

/// How many units of an opened scan's work its clones have claimed, out of how many in all.
///
/// A unit is what one clone claims at a time, such as a split of a layout scan. Claims only grow,
/// and `claimed` never exceeds `total`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Claims {
    /// The units that clones have claimed so far.
    pub claimed: usize,
    /// The units of the scan in all.
    pub total: usize,
}

/// The arrays of an opened scan, which host threads read by each polling a clone.
///
/// A clone is another worker over the same remaining work, like a clone of a channel's receiver:
/// the clones take work until none is left, so together they return every row of the scan
/// exactly once, in no particular order across clones. The number of clones a host polls is its
/// parallelism. The work may run as tasks on the session's runtime, which the host owns, and
/// nothing starts threads of its own. A host whose runtime has no threads of its own, such
/// as a `CurrentThreadRuntime`, must therefore drive that runtime while it waits on a clone, for
/// example with `block_on`. Dropping a clone before it ends gives up any work it had taken, so a
/// host drops a clone early only when it abandons the scan.
pub struct ScanStream {
    dtype: DType,
    new_worker: Arc<NewWorker>,
    claims: Option<Arc<ReadClaims>>,
    worker: Option<BoxStream<'static, VortexResult<ArrayRef>>>,
}

impl ScanStream {
    /// A stream whose clones each poll a worker from `new_worker`.
    ///
    /// The workers that `new_worker` returns must share one queue of work, so that together
    /// they return every row exactly once. A clone starts its worker when it is first polled.
    pub fn new(
        dtype: DType,
        new_worker: impl Fn() -> BoxStream<'static, VortexResult<ArrayRef>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            dtype,
            new_worker: Arc::new(new_worker),
            claims: None,
            worker: None,
        }
    }

    /// This stream, reporting through `claims` how many units of its work its clones have
    /// claimed, out of how many in all. `claims` must never block.
    pub fn with_claims(mut self, claims: impl Fn() -> Claims + Send + Sync + 'static) -> Self {
        self.claims = Some(Arc::new(claims));
        self
    }

    /// How many units of the scan's work its clones have claimed, out of how many in all, or
    /// `None` when the source does not know its total up front, as for a stream from
    /// [`Self::from_partitions`]. Never blocks.
    ///
    /// A host derives its progress from this. It also bounds how many clones a host needs: a
    /// host that hands out clones under its own lock may stop once it has handed out `total`
    /// clones, or once `claimed` reaches `total`, because the clones already handed out keep
    /// claiming units until none is left and so drain the rest. A clone handed out after that
    /// would find nothing to claim. When this is `None` the host cannot bound its clones this
    /// way, and learns that no work is left only when a clone ends.
    pub fn claims(&self) -> Option<Claims> {
        self.claims.as_ref().map(|claims| claims())
    }

    /// A stream whose clones take the partitions of `scan` one at a time.
    ///
    /// A source that has no queue of work of its own can open into this one.
    pub fn from_partitions(scan: DataSourceScanRef) -> Self {
        let dtype = scan.dtype().clone();
        let partitions = Arc::new(Mutex::new(scan.partitions()));
        Self::new(dtype, move || {
            let partitions = stream::unfold(Arc::clone(&partitions), |shared| async move {
                let next = shared.lock().await.next().await?;
                Some((next, shared))
            });
            partitions
                .flat_map(
                    |partition| match partition.and_then(|partition| partition.execute()) {
                        Ok(arrays) => arrays.left_stream(),
                        Err(err) => stream::once(async move { Err(err) }).right_stream(),
                    },
                )
                .boxed()
        })
    }
}

impl Clone for ScanStream {
    fn clone(&self) -> Self {
        vortex_io::diag::STREAM_CLONES.inc();
        Self {
            dtype: self.dtype.clone(),
            new_worker: Arc::clone(&self.new_worker),
            claims: self.claims.clone(),
            worker: None,
        }
    }
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

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let new_worker = &*this.new_worker;
        this.worker
            .get_or_insert_with(new_worker)
            .poll_next_unpin(cx)
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
    use vortex_array::expr::stats::Precision;
    use vortex_array::stream::ArrayStreamAdapter;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::stream::SendableArrayStream;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

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

    /// A scan that has no queue of work of its own, so it opens into [`ScanStream::from_partitions`].
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

        fn open(self: Box<Self>) -> VortexResult<ScanStream> {
            Ok(ScanStream::from_partitions(self))
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
    fn clones_share_partitions() -> VortexResult<()> {
        let stream = Box::new(PartitionOnlyScan {
            rows: vec![2, 3, 4, 5],
        })
        .open()?;
        assert_eq!(total_rows([stream.clone(), stream.clone(), stream])?, 14);
        Ok(())
    }
}

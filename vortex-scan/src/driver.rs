// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pull-driven execution of the Vortex Scan API without an engine or runtime dependency.
//!
//! ```no_run
//! # use std::num::NonZeroUsize;
//! # use futures::TryStreamExt;
//! # use vortex_error::VortexResult;
//! # use vortex_scan::{DataSource, ScanRequest};
//! # use vortex_scan::driver::ScanDriver;
//! # async fn read(source: &dyn DataSource) -> VortexResult<()> {
//! let request = ScanRequest::default();
//! let scan = source.scan(request.clone()).await?;
//! let mut arrays = ScanDriver::new(scan, &request)
//!     .into_stream(NonZeroUsize::new(4).unwrap());
//! while let Some(array) = arrays.try_next().await? {
//!     // Consume encoded Vortex arrays or convert them for the host engine.
//! }
//! # Ok(())
//! # }
//! ```

use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::Stream;
use futures::StreamExt as _;
use futures::TryStreamExt as _;
use futures::lock::Mutex;
use futures::stream;
use futures::stream::BoxStream;
use futures::task::AtomicWaker;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::DataSourceScanRef;
use crate::PartitionStream;
use crate::ScanRequest;

/// Drives one Scan API scan as independently polled array streams.
///
/// The driver spawns no tasks. Partitions own their I/O and CPU scheduling. Ordered scans
/// concatenate partitions in enumeration order; this does not merge overlapping sorted keys.
pub struct ScanDriver {
    scan: DataSourceScanRef,
    ordered: bool,
    limit: Option<u64>,
}

impl ScanDriver {
    /// Drives `scan` using the ordering and limit of the request that created it.
    pub fn new(scan: DataSourceScanRef, request: &ScanRequest) -> Self {
        Self {
            scan,
            ordered: request.ordered,
            limit: request.limit,
        }
    }

    /// Returns one lazy stream, executing at most `read_ahead` unordered partitions at once.
    pub fn into_stream(self, read_ahead: NonZeroUsize) -> SendableArrayStream {
        self.into_streams(NonZeroUsize::MIN, read_ahead)
            .pop()
            .vortex_expect("one output stream")
    }

    /// Distributes partitions among `outputs` lazy streams, with at most `read_ahead` active
    /// partitions per output. Consume every output concurrently for throughput.
    ///
    /// Ordered and limited scans return one output. Ordered scans execute partitions in sequence;
    /// limits trim the combined output exactly. Dropping an unfinished output or encountering an
    /// error cancels the execution and wakes sibling outputs, which report cancellation once.
    pub fn into_streams(
        self,
        outputs: NonZeroUsize,
        read_ahead: NonZeroUsize,
    ) -> Vec<SendableArrayStream> {
        let count = if self.ordered || self.limit.is_some() {
            1
        } else {
            outputs.get()
        };
        let dtype = self.scan.dtype().clone();
        let shared = Arc::new(Execution {
            partitions: Mutex::new(Some(self.scan.partitions())),
            cancelled: AtomicBool::new(false),
            waiters: (0..count).map(|_| AtomicWaker::new()).collect(),
        });
        (0..count)
            .map(|index| {
                let partitions: PartitionStream =
                    stream::try_unfold(Arc::clone(&shared), |shared| async move {
                        let partition = {
                            let mut source = shared.partitions.lock().await;
                            let next = match source.as_mut() {
                                Some(source) => source.try_next().await?,
                                None => None,
                            };
                            if next.is_none() {
                                source.take();
                            }
                            next
                        };
                        Ok(partition.map(|partition| (partition, shared)))
                    })
                    .boxed();
                let streams = partitions.map(|partition| partition?.execute());
                let arrays = if self.ordered {
                    streams.try_flatten().boxed()
                } else {
                    streams
                        .try_flatten_unordered(Some(read_ahead.get()))
                        .boxed()
                };
                Box::pin(Output {
                    dtype: dtype.clone(),
                    arrays: Some(arrays),
                    shared: Arc::clone(&shared),
                    index,
                    remaining: self.limit,
                }) as SendableArrayStream
            })
            .collect()
    }
}

struct Execution {
    partitions: Mutex<Option<PartitionStream>>,
    cancelled: AtomicBool,
    waiters: Vec<AtomicWaker>,
}

impl Execution {
    fn cancel(&self) {
        if self.cancelled.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Some(mut partitions) = self.partitions.try_lock() {
            partitions.take();
        }
        for waiter in &self.waiters {
            waiter.wake();
        }
    }
}

struct Output {
    dtype: DType,
    arrays: Option<BoxStream<'static, VortexResult<ArrayRef>>>,
    shared: Arc<Execution>,
    index: usize,
    remaining: Option<u64>,
}

impl Stream for Output {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.arrays.is_none() {
            return Poll::Ready(None);
        }
        self.shared.waiters[self.index].register(cx.waker());
        if self.shared.cancelled.load(Ordering::Acquire) {
            self.arrays.take();
            return Poll::Ready(Some(Err(vortex_err!("scan execution cancelled"))));
        }
        if self.remaining == Some(0) {
            self.arrays.take();
            self.shared.cancel();
            return Poll::Ready(None);
        }
        let result = self
            .arrays
            .as_mut()
            .vortex_expect("unfinished output")
            .poll_next_unpin(cx)
            .map(|next| next.map(|result| result.and_then(|array| self.trim(array))));
        match result {
            Poll::Ready(Some(Ok(array))) => {
                if self.remaining == Some(0) {
                    self.arrays.take();
                    self.shared.cancel();
                }
                Poll::Ready(Some(Ok(array)))
            }
            Poll::Ready(None) => {
                self.arrays.take();
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(error))) => {
                self.arrays.take();
                self.shared.cancel();
                Poll::Ready(Some(Err(error)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Output {
    fn trim(&mut self, array: ArrayRef) -> VortexResult<ArrayRef> {
        let Some(remaining) = &mut self.remaining else {
            return Ok(array);
        };
        let length = array
            .len()
            .min(usize::try_from(*remaining).unwrap_or(usize::MAX));
        let array = if length < array.len() {
            array.slice(0..length)?
        } else {
            array
        };
        *remaining -= length as u64;
        Ok(array)
    }
}

impl ArrayStream for Output {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        self.shared.waiters[self.index].take();
        if self.arrays.is_some() {
            self.shared.cancel();
        }
    }
}

#[cfg(test)]
mod tests;

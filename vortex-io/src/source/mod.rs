// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod file;

use futures::Stream;
use futures_util::{FutureExt, StreamExt};
use std::any::Any;
use std::ops::{Deref, Range};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::task::{Context, Poll, ready};
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};

/// An object capable of serving I/O requests tied to a specific file.
///
/// All underlying sources are wrapped in this struct to provide a common interface for submitting
/// zero-allocation async I/O requests. (Alternatives would require us to return a boxed future,
/// forcing an allocation for every I/O request.)
#[derive(Clone)]
pub struct IoSource(Arc<Inner>);
struct Inner {
    /// A bounded channel for buffering I/O requests.
    requests: flume::Sender<IoSourceRequest<dyn Any + Send + Sync>>,
    /// A counter tracking the total size of the requested buffers in the request queue.
    requests_size: AtomicU64,
    /// The data to attach to each IoRequest.
    data: Arc<dyn Any + Send + Sync>,
}

// I think we probably need another concept. For example:
//  * Source - a future that processes a stream of requests that ultimately controls access
//    to the underlying shared resource.
//  * File - a wrapper around a Source that attaches a piece of Arc'd data to each IoRequest. This
//    enables the source to narrow the scope to a particular file.
//
// But what about for file-level data that cannot be arc'd? For example, a Tokio file produces
// non-send futures.

impl IoSource {
    /// Create a new instance of [`IoSource`] based on the given [`IoDriver`].
    pub fn try_new<D: IoDriver>(driver: D, data: Arc<D::Data>) -> VortexResult<Self> {
        // TODO(ngates): for now, we bound the number of requests. Instead, we should bound
        //  by the total size of the requests.
        let (send, recv) = flume::bounded::<IoSourceRequest<dyn Any + Send + Sync>>(64);

        // But we spawn the driver for every instance of the source... not ideal.
        driver.spawn(
            recv.into_stream()
                .map(|req| req.downcast::<D::Data>())
                .boxed(),
        )?;

        Ok(Self(Arc::new(Inner {
            requests: send,
            requests_size: AtomicU64::new(0),
            data,
        })))
    }

    /// Attempt to enqueue the request to the I/O source.
    /// Returns `Poll::Pending` if the I/O source's request queue is full.
    ///
    // NOTE(ngates): this API helps us avoid returning a boxed future, and therefore requiring
    //  an allocation for every I/O request. Instead, the caller can hold onto the local callback
    //  future of the IoRequest and poll it as needed.
    pub fn read(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> impl Future<Output = VortexResult<ByteBuffer>> + Send + '_ {
        // TODO(ngates): we should claim weight in the requests_size counter here before
        //  submitting to the queue. For now, we're bounded by the length of the bounded channel.
        // TODO(ngates): wrap the request to track metrics about the throughput / latency.
        let (request, fut) = IoRequest::new(offset, length, alignment);

        let src = self.0.clone();
        async move {
            if let Err(e) = src
                .requests
                .send_async(IoSourceRequest {
                    request,
                    data: self.0.data.clone(),
                })
                .await
            {
                vortex_bail!("IoSource sender dropped while in flight: {e}");
            }
            fut.await
        }
    }
}

pub trait IoDriver {
    type Data: Any + Send + Sync;

    fn spawn(
        &self,
        requests: impl Stream<Item = IoSourceRequest<Self::Data>> + Send + 'static,
    ) -> VortexResult<()>;
}

pub struct IoSourceRequest<Data: ?Sized> {
    request: IoRequest,
    data: Arc<Data>,
}

impl<Data> IoSourceRequest<Data> {
    pub fn data(&self) -> &Data {
        &self.data
    }

    /// Resolve the [`IoRequest`].
    pub fn resolve(self, result: VortexResult<ByteBuffer>) {
        self.request.resolve(result);
    }
}

impl<D> Deref for IoSourceRequest<D> {
    type Target = IoRequest;

    fn deref(&self) -> &Self::Target {
        &self.request
    }
}

impl IoSourceRequest<dyn Any + Send + Sync> {
    fn downcast<D: Any + Send + Sync>(self) -> IoSourceRequest<D> {
        IoSourceRequest {
            request: self.request,
            data: self
                .data
                .downcast()
                .map_err(|_| vortex_err!("invalid downcast"))
                .vortex_expect("invalid downcast"),
        }
    }
}

/// A generalized read trait for Vortex.
pub trait Read {
    fn read(&self, offset: u64, length: usize, alignment: Alignment) -> ReadFuture;
}

/// A future returned from the Vortex [`Read`] trait. By specifying a concrete type, we can
/// typically avoid heap-allocating boxed futures.
pub struct ReadFuture(oneshot::Receiver<VortexResult<ByteBuffer>>);

impl Future for ReadFuture {
    type Output = VortexResult<ByteBuffer>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match ready!(self.0.poll_unpin(cx)) {
            Ok(result) => Poll::Ready(result),
            Err(e) => Poll::Ready(Err(vortex_err!("Receive error {e}"))),
        }
    }
}

/// An [`IoRequest`] that can be submitted to an [`IoDriver`].
pub struct IoRequest {
    pub offset: u64,
    pub length: usize,
    pub alignment: Alignment,
    callback: oneshot::Sender<VortexResult<ByteBuffer>>,
}

impl IoRequest {
    pub fn new(
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> (
        Self,
        impl Future<Output = VortexResult<ByteBuffer>> + Send + 'static,
    ) {
        let (send, recv) = oneshot::channel();
        let req = Self {
            offset,
            length,
            alignment,
            callback: send,
        };

        let recv = async move {
            recv.await
                .map_err(|e| vortex_err!("Receive error {e}"))
                .flatten()
        };

        (req, recv)
    }

    /// Returns a u64 read range for this request.
    pub fn range(&self) -> Range<u64> {
        self.offset..self.offset + self.length as u64
    }

    /// Resolve the [`IoRequest`].
    pub fn resolve(self, result: VortexResult<ByteBuffer>) {
        if let Err(e) = self.callback.send(result) {
            log::trace!("IoRequest sender dropped while in flight: {e}");
        }
    }
}

pub trait IntoIoSource {
    fn into_io_source(self) -> VortexResult<IoSource>;
}

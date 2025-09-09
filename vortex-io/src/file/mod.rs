// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod buffer;
mod driver;
#[cfg(feature = "object_store")]
pub mod object_store;
mod request;
mod source;
#[cfg(not(target_arch = "wasm32"))]
mod std_file;

use std::fmt;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, ready};

pub(crate) use driver::*;
use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, TryFutureExt};
pub use request::*;
pub use source::*;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{SharedVortexResult, VortexError, VortexResult, vortex_err};

/// A handle to an open file that can be read using a Vortex runtime.
///
/// ## Coalescing and Pre-fetching
///
/// It is important to understand the semantics of the read futures returned by a [`FileRead`].
/// Under the hood, each [`FileRead`] is backed by a stream that services read requests by
/// applying coalescing and concurrency constraints.
///
/// Each read future has four states:
/// * `registered` - the read future has been created, but not yet polled.
/// * `requested` - the read future has been polled.
/// * `in-flight` - the read request has been sent to the underlying storage system.
/// * `resolved` - the read future has completed and resolved a result.
///
/// When a read request is `registered`, it will not itself trigger any I/O, but is eligible to
/// be coalesced with other requests.
///
/// If a [`ReadFuture`] is dropped, it will be canceled if possible. This depends on the current
/// state of the request, as well as whether the underlying storage system supports cancellation.
///
/// I/O requests will be processed in the order they are `registered`, however coalescing may mean
/// other registered requests are lumped together into a single I/O operation.
#[derive(Clone)]
pub struct FileRead<'rt> {
    /// Human-readable descriptor for the file, typically its URI.
    uri: Arc<str>,
    /// A shared future that resolves to the size of the file.
    size: Shared<BoxFuture<'static, SharedVortexResult<u64>>>,
    /// A queue for sending read request events to the I/O stream.
    events: kanal::Sender<ReadEvent>,
    /// The next read request ID.
    next_id: Arc<AtomicUsize>,
    /// Lifetime that ties the file handle to the runtime it was opened on.
    _rt: PhantomData<&'rt ()>,
}

impl Debug for FileRead<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileHandle")
            .field("uri", &self.uri)
            .finish()
    }
}

impl Display for FileRead<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.uri)
    }
}

impl<'rt> FileRead<'rt> {
    pub(crate) fn new(
        uri: Arc<str>,
        size: BoxFuture<'static, VortexResult<u64>>,
        send: kanal::Sender<ReadEvent>,
    ) -> Self {
        Self {
            uri,
            size: size.map_err(Arc::new).boxed().shared(),
            events: send,
            next_id: Arc::new(AtomicUsize::new(0)),
            _rt: Default::default(),
        }
    }

    /// The URI of the file.
    pub fn uri(&self) -> &Arc<str> {
        &self.uri
    }

    /// Returns the size of the file in bytes.
    pub fn size(&self) -> impl Future<Output = VortexResult<u64>> + Send + 'rt {
        self.size.clone().map_err(VortexError::from)
    }

    /// Submits a read request for the specified byte range and alignment.
    pub fn read(&self, offset: u64, length: usize, alignment: Alignment) -> ReadFuture<'rt> {
        let (send, recv) = oneshot::channel();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let event = ReadEvent::Request(ReadRequest {
            id,
            offset,
            length,
            alignment,
            callback: send,
        });

        // If we fail to submit the event, we create a ReadFuture that has already failed.
        if let Err(e) = self.events.send(event) {
            let (send, recv) = oneshot::channel();
            let _ = send.send(Err(vortex_err!("Failed to submit read request: {e}")));
            return ReadFuture {
                id,
                recv,
                polled: false,
                events: self.events.clone(),
                _rt: PhantomData,
            };
        }

        ReadFuture {
            id,
            recv,
            polled: false,
            events: self.events.clone(),
            _rt: PhantomData,
        }
    }
}

#[derive(Debug)]
pub(crate) enum ReadEvent {
    Request(ReadRequest),
    Polled(RequestId),
    Dropped(RequestId),
}

/// A future that resolves a read request from a [`FileRead`].
///
/// See the documentation for [`FileRead`] for details on coalescing and pre-fetching.
/// If dropped, the read request will be canceled where possible.
pub struct ReadFuture<'rt> {
    id: usize,
    recv: oneshot::Receiver<VortexResult<ByteBuffer>>,
    polled: bool,
    events: kanal::Sender<ReadEvent>,
    _rt: PhantomData<&'rt ()>,
}

impl Future for ReadFuture<'_> {
    type Output = VortexResult<ByteBuffer>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.polled {
            self.polled = true;
            // Notify the I/O stream that this request has been polled.
            if let Err(e) = self.events.send(ReadEvent::Polled(self.id)) {
                return Poll::Ready(Err(vortex_err!("ReadRequest dropped by runtime: {e}")));
            }
        }

        match ready!(self.recv.poll_unpin(cx)) {
            Ok(result) => Poll::Ready(result),
            Err(e) => Poll::Ready(Err(vortex_err!("ReadRequest dropped by runtime: {e}"))),
        }
    }
}

impl Drop for ReadFuture<'_> {
    fn drop(&mut self) {
        // When the FileHandle is dropped, we can send a shutdown event to the I/O stream.
        // If the I/O stream has already been dropped, this will fail silently.
        let _ = self.events.send(ReadEvent::Dropped(self.id));
    }
}

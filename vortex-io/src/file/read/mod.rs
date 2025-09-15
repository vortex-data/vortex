// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod request;
mod source;

use std::fmt;
use std::fmt::{Debug, Display};
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{ready, Context, Poll};

use async_trait::async_trait;
use futures::channel::mpsc;
use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, TryFutureExt};
pub use request::*;
pub use source::*;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{vortex_err, SharedVortexResult, VortexExpect, VortexResult};

use crate::VortexReadAt;

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
pub struct FileRead {
    /// Human-readable descriptor for the file, typically its URI.
    uri: Arc<str>,
    /// A shared future that resolves to the size of the file.
    size: Shared<BoxFuture<'static, SharedVortexResult<u64>>>,
    /// A queue for sending read request events to the I/O stream.
    events: mpsc::UnboundedSender<ReadEvent>,
    /// The next read request ID.
    next_id: Arc<AtomicUsize>,
}

impl Debug for FileRead {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileHandle")
            .field("uri", &self.uri)
            .finish()
    }
}

impl Display for FileRead {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.uri)
    }
}

impl FileRead {
    pub(crate) fn new(
        uri: Arc<str>,
        size: BoxFuture<'static, VortexResult<u64>>,
        send: mpsc::UnboundedSender<ReadEvent>,
    ) -> Self {
        Self {
            uri,
            size: size.map_err(Arc::new).boxed().shared(),
            events: send,
            next_id: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// The URI of the file.
    pub fn uri(&self) -> &Arc<str> {
        &self.uri
    }

    /// Submits a read request for the specified byte range and alignment.
    pub fn read(&self, offset: u64, length: usize, alignment: Alignment) -> ReadFuture {
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
        if let Err(e) = self.events.unbounded_send(event) {
            let (send, recv) = oneshot::channel();
            let _ = send.send(Err(vortex_err!("Failed to submit read request: {e}")));
            return ReadFuture {
                id,
                recv,
                polled: false,
                events: self.events.clone(),
            };
        }

        ReadFuture {
            id,
            recv,
            polled: false,
            events: self.events.clone(),
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
pub struct ReadFuture {
    id: usize,
    recv: oneshot::Receiver<VortexResult<ByteBuffer>>,
    polled: bool,
    events: mpsc::UnboundedSender<ReadEvent>,
}

impl Future for ReadFuture {
    type Output = VortexResult<ByteBuffer>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.polled {
            self.polled = true;
            // Notify the I/O stream that this request has been polled.
            if let Err(e) = self.events.unbounded_send(ReadEvent::Polled(self.id)) {
                return Poll::Ready(Err(vortex_err!("ReadRequest dropped by runtime: {e}")));
            }
        }

        match ready!(self.recv.poll_unpin(cx)) {
            Ok(result) => Poll::Ready(result),
            Err(e) => Poll::Ready(Err(vortex_err!("ReadRequest dropped by runtime: {e}"))),
        }
    }
}

impl Drop for ReadFuture {
    fn drop(&mut self) {
        // When the FileHandle is dropped, we can send a shutdown event to the I/O stream.
        // If the I/O stream has already been dropped, this will fail silently.
        let _ = self.events.unbounded_send(ReadEvent::Dropped(self.id));
    }
}

#[async_trait]
impl VortexReadAt for FileRead {
    fn read_byte_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> BoxFuture<'static, std::io::Result<ByteBuffer>> {
        let length = usize::try_from(range.end - range.start)
            .vortex_expect("Read range too large for usize");
        self.read(range.start, length, alignment)
            .map_err(|e| std::io::Error::other(format!("Vortex read error: {e}")))
            .boxed()
    }

    async fn size(&self) -> std::io::Result<u64> {
        self.size
            .clone()
            .await
            .map_err(|e| std::io::Error::other(format!("Vortex read error: {e}")))
    }
}

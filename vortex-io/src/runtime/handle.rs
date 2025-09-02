// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::kanal_ext::KanalExt;
use crate::runtime::io::IoRequestStream;
use crate::runtime::source::IoSource;
use crate::runtime::{CpuTask, IoTask, ReadRequest, Runtime};
use async_compat::Compat;
use futures::future::{BoxFuture, Shared};
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt, TryFutureExt};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{ready, Context, Poll};
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{
    vortex_err, vortex_panic, SharedVortexResult, VortexError, VortexExpect, VortexResult,
};

/// Represents a handle to a Vortex runtime that can be used to enqueue CPU- or I/O-bound tasks.
#[derive(Clone)]
pub struct Handle<'rt>(pub(crate) Arc<dyn Runtime<'rt> + 'rt>);

impl Handle<'static> {
    // FIXME(ngates): remove this!
    pub fn no_op() -> Self {
        struct NoOp;

        impl Runtime<'static> for NoOp {
            fn spawn_scheduling(&self, _fut: BoxFuture<'static, ()>) {
                vortex_panic!("NoOp runtime cannot spawn tasks");
            }

            fn spawn_cpu(&self, _task: CpuTask) {
                vortex_panic!("NoOp runtime cannot spawn CPU tasks");
            }

            fn spawn_io(&self, _stream: BoxStream<'static, IoTask>, _concurrency: usize) {
                vortex_panic!("NoOp runtime cannot spawn I/O tasks");
            }
        }

        Self(Arc::new(NoOp))
    }
}

impl<'rt> Handle<'rt> {
    /// Spawn a new scheduling future onto the runtime.
    pub fn spawn<Fut, R>(&self, f: Fut) -> impl Future<Output = R> + use<'rt, Fut, R>
    where
        Fut: Future<Output = R> + Send + 'rt,
        R: Send + 'rt,
    {
        let (send, recv) = oneshot::channel();
        self.0.spawn_scheduling(
            async move {
                if let Err(e) = send.send(f.await) {
                    log::trace!("Failed to send task result: {e}");
                }
            }
            .boxed(),
        );
        async move {
            recv.await
                .map_err(|e| vortex_err!("Failed to await result, runtime dropped: {e}"))
                .vortex_expect("Failed to await result")
        }
    }

    /// Spawn a CPU-bound task for execution on the runtime.
    pub fn spawn_cpu<F, R>(&self, f: F) -> impl Future<Output = R> + Send + 'rt
    where
        // Unlike scheduling futures, the CPU task should have a static lifetime because it
        // doesn't need to access to handle to spawn more work.
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // TODO(ngates): we want a droppable handle for this.
        let (send, recv) = oneshot::channel();
        self.0.spawn_cpu(CpuTask {
            runnable: Box::new(move || {
                let _ = send.send(f());
            }),
        });
        async move {
            recv.await
                .map_err(|e| vortex_err!("Task cancelled: {e}"))
                .vortex_expect("Task cancelled")
        }
    }

    /// Open an [`IoSource`] for reading using this runtime.
    ///
    /// This sets up an I/O request queue that is processed by the runtime. See the documentation
    /// for [`FileIo`] to understand the lifecycle of a read request.
    pub fn open<S: IoSource>(&self, source: S) -> FileIo<'rt> {
        let source = Arc::new(source);
        let (send, recv) = kanal::unbounded();

        // Construct the size future in case we need it.
        let size = Compat::new(source.size().map_err(Arc::new))
            .boxed()
            .shared();

        let concurrency = source.concurrency();
        let name = source.name();

        let stream = recv.to_async().into_stream().boxed();
        let stream = IoRequestStream::new(stream, source.coalescing_window())
            .map(move |req| IoTask::new(source.clone(), req))
            .boxed();

        self.0.spawn_io(stream, concurrency);

        FileIo {
            name,
            size,
            send,
            _phantom: Default::default(),
        }
    }
}

/// A file that can be read from using a Vortex runtime.
///
/// This essentially provides a wrapper to bind a handle to a read interface. It is optional, but
/// should be used carefully because the subsequent read operations must be driven on the same
/// runtime.
#[derive(Clone)]
pub struct FileIo<'rt> {
    name: String,
    size: Shared<BoxFuture<'static, SharedVortexResult<u64>>>,
    send: kanal::Sender<ReadRequest>,
    _phantom: PhantomData<&'rt ()>,
}

impl FileIo<'_> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn read(&self, offset: u64, length: usize, alignment: Alignment) -> Read {
        let (read, callback) = Read::new();
        if let Err(e) = self.send.send(ReadRequest {
            offset,
            length,
            alignment,
            completion: callback,
        }) {
            vortex_panic!("Failed to send I/O task, runtime terminated: {e}");
        }
        read
    }

    pub fn size(&self) -> impl Future<Output = VortexResult<u64>> {
        self.size.clone().map_err(VortexError::from)
    }
}

/// A future representing an in-flight read operation.
///
/// If this [`Read`] object is dropped prior to the I/O operation being submitted, it **may** be
/// skipped by the runtime. If it has already been submitted, it will continue to completion.
pub struct Read {
    /// The oneshot receiver to get the result of the read operation.
    recv: oneshot::Receiver<VortexResult<ByteBuffer>>,
    /// Indicates whether this future has been polled.
    polled: Arc<AtomicBool>,
}

impl Read {
    pub(crate) fn new() -> (Self, ReadCompletion) {
        let (send, recv) = oneshot::channel();
        let polled = Arc::new(AtomicBool::new(false));
        (
            Read {
                recv,
                polled: polled.clone(),
            },
            ReadCompletion { send, polled },
        )
    }
}

/// A handle to complete a pending read operation.
pub(crate) struct ReadCompletion {
    /// The oneshot sender to complete the read operation.
    send: oneshot::Sender<VortexResult<ByteBuffer>>,
    /// Indicates whether the consumer end of the read has been polled.
    polled: Arc<AtomicBool>,
}

impl ReadCompletion {
    /// Returns true if the read has been canceled and the result will not be delivered.
    pub fn is_canceled(&self) -> bool {
        self.send.is_closed()
    }

    /// Returns true if the read operation has been polled.
    pub fn is_polled(&self) -> bool {
        self.polled.load(Ordering::Relaxed)
    }

    /// Complete the read operation with the given result.
    pub fn complete(self, result: VortexResult<ByteBuffer>) {
        if let Err(e) = self.send.send(result) {
            log::trace!("I/O request cancelled while in-flight: {e}");
        }
    }
}

impl Future for Read {
    type Output = VortexResult<ByteBuffer>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.polled.store(true, Ordering::Relaxed);
        match ready!(self.recv.poll_unpin(cx)) {
            Ok(result) => Poll::Ready(result),
            Err(e) => Poll::Ready(Err(vortex_err!(
                "Failed to read from file, IoTask dropped by runtime: {e}"
            ))),
        }
    }
}

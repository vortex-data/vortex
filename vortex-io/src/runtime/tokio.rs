// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, FileIoRequest, Handle, Runtime};
use futures::Stream;
use futures_util::future::BoxFuture;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use tokio::runtime::Handle as TokioHandle;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;

/// A Vortex runtime that drives all work on a provided Tokio runtime.
#[derive(Clone)]
pub struct TokioRuntime(Arc<TokioHandle>);

impl TokioRuntime {
    pub fn new(handle: TokioHandle) -> Self {
        TokioRuntime(Arc::new(handle))
    }
}

impl Default for TokioRuntime {
    fn default() -> Self {
        Self::new(TokioHandle::current())
    }
}

impl Runtime for TokioHandle {
    fn spawn_scheduling(&self, fut: BoxFuture<'static, ()>) {
        TokioHandle::spawn(self, fut);
    }

    fn spawn_cpu(&self, f: CpuTask) {
        // We spawn CPU tasks as if they were normal async tasks on the Tokio runtime.
        TokioHandle::spawn(self, async move { f.run() });
    }

    fn spawn_io(&self, f: FileIoRequest) {
        // FIXME(ngates): the API for a Runtime is dumb.
        // TokioHandle::spawn_blocking(self, move || {
        let mut buffer = ByteBufferMut::with_capacity_aligned(f.length, f.alignment);
        unsafe { buffer.set_len(f.length) };
        match f.file.read_exact_at(&mut buffer, f.offset) {
            Ok(()) => f.resolve(Ok(buffer.freeze())),
            Err(e) => f.resolve(Err(VortexError::from(e))),
        }
        // });
    }
}

impl TokioRuntime {
    /// Drive the given Vortex future on the underlying Tokio runtime.
    pub fn drive<F, Fut, R>(self, f: F) -> impl Future<Output = R> + Send + 'static
    where
        F: FnOnce(Handle) -> Fut,
        Fut: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        f(Handle(self.0))
    }

    /// Drive the given Vortex stream on the underlying Tokio runtime.
    pub fn drive_stream<F, S, R>(self, f: F) -> impl Stream<Item = R> + Send + 'static
    where
        F: FnOnce(Handle) -> S,
        S: Stream<Item = R> + Send + 'static,
        R: Send + 'static,
    {
        f(Handle(self.0))
    }
}

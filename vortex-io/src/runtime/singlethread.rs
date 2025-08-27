// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, Handle, IoTask, Runtime};
use async_compat::Compat;
use futures::executor::block_on;
use futures::future::BoxFuture;
use futures::stream::{BoxStream, LocalBoxStream};
use futures::Stream;
use futures::StreamExt;
use smol::LocalExecutor;
use std::rc::Rc;
use std::sync::Arc;
use vortex_error::vortex_panic;

/// A runtime that drives all work on the current thread.
///
/// Since the [`Handle`], and therefore [`Runtime`] implementation needs to be `Send` and `Sync`,
/// we cannot just `impl Runtime for LocalExecutor`. Instead, we create channels that the handle
/// can forward its work into, and we drive the resulting tasks on a [`LocalExecutor`] on the
/// calling thread.
// TODO(ngates): use a builder to configure whether I/O runs on a separate blocking pool or not.
pub struct SingleThreadRuntime {
    scheduling: flume::Sender<BoxFuture<'static, ()>>,
    cpu: flume::Sender<CpuTask>,
    io: flume::Sender<(BoxStream<'static, IoTask>, usize)>,
}

impl SingleThreadRuntime {
    fn new() -> (Self, Rc<LocalExecutor<'static>>) {
        let (scheduling_send, scheduling_recv) = flume::unbounded::<BoxFuture<'static, ()>>();
        let (cpu_send, cpu_recv) = flume::unbounded::<CpuTask>();
        let (io_send, io_recv) = flume::unbounded::<(BoxStream<'static, IoTask>, usize)>();

        let local = Rc::new(LocalExecutor::new());

        // Drive scheduling tasks.
        let local2 = local.clone();
        local
            .spawn(async move {
                while let Ok(fut) = scheduling_recv.recv_async().await {
                    local2.spawn(fut).detach();
                }
            })
            .detach();

        // Drive CPU tasks.
        let local2 = local.clone();
        local
            .spawn(async move {
                while let Ok(task) = cpu_recv.recv_async().await {
                    local2.spawn(async move { task.run() }).detach();
                }
            })
            .detach();

        // Drive I/O tasks.
        let local2 = local.clone();
        local
            .spawn(async move {
                while let Ok((stream, concurrency)) = io_recv.recv_async().await {
                    // NOTE(ngates): for now, we allow arbitrary Tokio I/O and therefore wrap up
                    //  the futures in a compatibility layer.
                    local2
                        .spawn(Compat::new(async move {
                            stream
                                .map(|task| task.run_local())
                                .buffer_unordered(concurrency)
                                .collect::<()>()
                                .await
                        }))
                        .detach();
                }
            })
            .detach();

        let this = Self {
            scheduling: scheduling_send,
            cpu: cpu_send,
            io: io_send,
        };

        (this, local)
    }

    /// Drive the given Vortex future on the underlying Tokio runtime.
    pub fn drive<F, Fut, R>(f: F) -> R
    where
        F: FnOnce(Handle) -> Fut,
        Fut: Future<Output = R>,
        R: Send + 'static,
    {
        let (rt, local) = Self::new();
        let fut = f(Handle(Arc::new(rt)));
        block_on(local.run(fut))
    }

    /// Drive the given Vortex stream on the underlying Tokio runtime.
    pub fn drive_stream<F, S, R>(f: F) -> impl Iterator<Item = R>
    where
        F: FnOnce(Handle) -> S,
        S: Stream<Item = R> + Unpin + 'static,
        R: Send + 'static,
    {
        let (rt, local) = Self::new();
        let stream = f(Handle(Arc::new(rt)));
        BlockingStream {
            executor: local,
            stream: stream.boxed_local(),
        }
    }
}

impl Runtime for SingleThreadRuntime {
    fn spawn_scheduling(&self, fut: BoxFuture<'static, ()>) {
        if let Err(e) = self.scheduling.send(fut) {
            vortex_panic!("Runtime dropped while scheduling task: {}", e);
        }
    }

    fn spawn_cpu(&self, task: CpuTask) {
        if let Err(e) = self.cpu.send(task) {
            vortex_panic!("Runtime dropped while scheduling CPU task: {}", e);
        }
    }

    fn spawn_io(&self, stream: BoxStream<'static, IoTask>, concurrency: usize) {
        if let Err(e) = self.io.send((stream, concurrency)) {
            vortex_panic!("Runtime dropped while scheduling I/O task: {}", e);
        }
    }
}

struct BlockingStream<T> {
    executor: Rc<LocalExecutor<'static>>,
    stream: LocalBoxStream<'static, T>,
}

impl<T> Iterator for BlockingStream<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let fut = self.stream.next();
        block_on(self.executor.run(fut))
    }
}

#[cfg(test)]
mod tests {
    use crate::runtime::singlethread::SingleThreadRuntime;
    use std::fs::File;
    use std::io::Write;
    use std::sync::Arc;
    use vortex_buffer::Alignment;

    #[test]
    fn test_oneshot() {
        {
            // First, we write some dummy data to a temporary file.
            let mut file = File::create("test.txt").unwrap();
            file.write_all(b"Hello, Vortex!").unwrap();
        }

        let buffer = SingleThreadRuntime::drive(|handle| {
            // Now we read from the file using the handle.
            let read = handle.open_file(Arc::new(File::open("test.txt").unwrap()));
            read.read(0, 14, Alignment::none())
        })
        .unwrap();

        assert_eq!(buffer.as_ref(), b"Hello, Vortex!");

        // Finally, we clean up the temporary file.
        std::fs::remove_file("test.txt").unwrap()
    }
}

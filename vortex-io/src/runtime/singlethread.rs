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
pub struct SingleThreadRuntime<'rt> {
    scheduling: flume::Sender<BoxFuture<'rt, ()>>,
    cpu: flume::Sender<CpuTask>,
    io: flume::Sender<(BoxStream<'rt, IoTask>, usize)>,
}

impl<'rt> SingleThreadRuntime<'rt> {
    fn new(local: Rc<LocalExecutor<'rt>>) -> Self {
        let (scheduling_send, scheduling_recv) = flume::unbounded::<BoxFuture<'rt, ()>>();
        let (cpu_send, cpu_recv) = flume::unbounded::<CpuTask>();
        let (io_send, io_recv) = flume::unbounded::<(BoxStream<'rt, IoTask>, usize)>();

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

        Self {
            scheduling: scheduling_send,
            cpu: cpu_send,
            io: io_send,
        }
    }

    /// Drive the given Vortex future on the underlying Tokio runtime.
    pub fn drive<F, Fut, R>(f: F) -> R
    where
        F: FnOnce(Handle<'rt>) -> Fut,
        Fut: Future<Output = R> + 'rt,
        R: Send + 'rt,
    {
        let executor = Rc::new(LocalExecutor::new());
        let rt = Arc::new(SingleThreadRuntime::new(executor.clone()));
        let fut = f(Handle(rt));
        block_on(executor.run(fut))
    }

    /// Drive the given Vortex stream on the underlying Tokio runtime.
    pub fn drive_stream<F, S, R>(f: F) -> impl Iterator<Item = R>
    where
        F: FnOnce(Handle<'rt>) -> S,
        S: Stream<Item = R> + Unpin + 'rt,
        R: Send + 'rt,
    {
        // Create a new static executor.
        let executor = Rc::new(LocalExecutor::new());
        let rt = Arc::new(SingleThreadRuntime::new(executor.clone()));
        let stream = f(Handle(rt));

        // SAFETY: The stream contains references to `rt` with lifetime 'rt.
        // We're transmuting this to 'static, which is sound because:
        // 1. Both `rt` and `stream` will be moved into BlockingStream
        // 2. BlockingStream will drop them in the correct order (stream first, then rt)
        // 3. The stream will never outlive the runtime it references
        let executor: Rc<LocalExecutor> = unsafe {
            std::mem::transmute::<Rc<LocalExecutor<'_>>, Rc<LocalExecutor<'static>>>(executor)
        };
        let stream: LocalBoxStream<'static, R> = unsafe {
            std::mem::transmute::<LocalBoxStream<'_, R>, LocalBoxStream<'static, R>>(
                stream.boxed_local(),
            )
        };

        BlockingStream { executor, stream }
    }
}

impl<'rt> Runtime<'rt> for SingleThreadRuntime<'rt> {
    fn spawn_scheduling(&self, fut: BoxFuture<'rt, ()>) {
        if let Err(e) = self.scheduling.send(fut) {
            vortex_panic!("Runtime dropped while scheduling task: {}", e);
        }
    }

    fn spawn_cpu(&self, task: CpuTask) {
        if let Err(e) = self.cpu.send(task) {
            vortex_panic!("Runtime dropped while scheduling CPU task: {}", e);
        }
    }

    fn spawn_io(&self, stream: BoxStream<'rt, IoTask>, concurrency: usize) {
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

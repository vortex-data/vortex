// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, Handle, IoTask, Runtime};
use futures::executor::block_on;
use futures::future::BoxFuture;
use futures::stream::{BoxStream, LocalBoxStream};
use futures::Stream;
use futures::StreamExt;
use smol::Executor;
use std::sync::Arc;

/// A runtime that drives all work on the current thread.
// FIXME(ngates): use a builder to configure whether I/O runs on a separate blocking pool or not.
pub struct SingleThreadRuntime;

impl SingleThreadRuntime {
    /// Drive the given Vortex future on the underlying Tokio runtime.
    pub fn drive<F, Fut, R>(f: F) -> R
    where
        F: FnOnce(Handle) -> Fut,
        Fut: Future<Output = R>,
        R: Send + 'static,
    {
        let ex = Arc::new(Executor::new());
        let fut = f(Handle(ex.clone()));
        block_on(ex.run(fut))
    }

    /// Drive the given Vortex stream on the underlying Tokio runtime.
    pub fn drive_stream<F, S, R>(f: F) -> impl Iterator<Item = R>
    where
        F: FnOnce(Handle) -> S,
        S: Stream<Item = R> + Unpin + 'static,
        R: Send + 'static,
    {
        let executor = Arc::new(Executor::new());
        let stream = f(Handle(executor.clone()));
        BlockingStream {
            executor,
            stream: stream.boxed_local(),
        }
    }
}

impl Runtime for Executor<'static> {
    fn spawn_scheduling(&self, fut: BoxFuture<'static, ()>) {
        self.spawn(fut).detach()
    }

    fn spawn_cpu(&self, task: CpuTask) {
        self.spawn(async move { task.run() }).detach();
    }

    fn spawn_io(&self, mut stream: BoxStream<'static, IoTask>) {
        self.spawn(async move {
            while let Some(task) = stream.next().await {
                task.run().await;
            }
        })
        .detach()
    }
}

struct BlockingStream<T> {
    executor: Arc<Executor<'static>>,
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

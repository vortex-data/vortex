// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, FileIoRequest, Handle, Runtime};
use futures::executor::{block_on, block_on_stream};
use futures::Stream;
use futures_util::future::BoxFuture;
use smol::Executor;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;

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
        block_on(f(Handle(Arc::new(Executor::new()))))
    }

    /// Drive the given Vortex stream on the underlying Tokio runtime.
    pub fn drive_stream<F, S, R>(f: F) -> impl Iterator<Item = R>
    where
        F: FnOnce(Handle) -> S,
        S: Stream<Item = R> + Unpin,
        R: Send + 'static,
    {
        block_on_stream(f(Handle(Arc::new(Executor::new()))))
    }
}

impl Runtime for Executor<'static> {
    fn spawn_scheduling(&self, fut: BoxFuture<'static, ()>) {
        self.spawn(fut).detach()
    }

    fn spawn_cpu(&self, task: CpuTask) {
        self.spawn(async move { task.run() }).detach();
    }

    fn spawn_io(&self, f: FileIoRequest) {
        smol::unblock(move || {
            let mut buffer = ByteBufferMut::with_capacity_aligned(f.length, f.alignment);
            unsafe { buffer.set_len(f.length) };
            match f.file.read_exact_at(&mut buffer, f.offset) {
                Ok(()) => f.resolve(Ok(buffer.freeze())),
                Err(e) => f.resolve(Err(VortexError::from(e))),
            }
        })
        .detach();
    }
}

#[cfg(test)]
mod tests {
    use crate::runtime::singlethread::SingleThreadRuntime;
    use crate::source::FileIo;
    use std::io::Write;
    use vortex_buffer::Alignment;

    #[test]
    fn test_oneshot() {
        {
            // First, we write some dummy data to a temporary file.
            let mut file = std::fs::File::create("test.txt").unwrap();
            file.write_all(b"Hello, Vortex!").unwrap();
        }

        let buffer = SingleThreadRuntime::drive(|handle| {
            // Now we read from the file using the handle.
            let read = FileIo::try_new("test.txt")
                .expect("Failed to create IoSource")
                .open(&handle);

            read.read(0, 14, Alignment::none())
        })
        .unwrap();

        assert_eq!(buffer.as_ref(), b"Hello, Vortex!");

        // Finally, we clean up the temporary file.
        std::fs::remove_file("test.txt").unwrap()
    }
}

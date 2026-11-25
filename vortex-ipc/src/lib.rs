// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex IPC messages and associated readers and writers.
//!
//! Vortex provides an IPC messaging format to exchange array data over a streaming
//! interface. The format emits message headers in FlatBuffer format, along with their
//! data buffers.
//!
//! This crate provides both in-memory message representations for holding IPC messages
//! before/after serialization, and streaming readers and writers that sit on top
//! of any type implementing `VortexRead` or `VortexWrite` respectively.

pub mod flatbuffers;
pub mod iterator;
pub mod messages;
pub mod stream;

#[cfg(test)]
mod test {
    use std::error::Error as _;
    use std::io;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::task::Context;
    use std::task::Poll;

    use futures::AsyncRead;
    use vortex_array::aggregate_fn::session::AggregateFnSession;
    use vortex_array::dtype::session::DTypeSession;
    use vortex_array::optimizer::kernels::KernelSession;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexError;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    pub(crate) static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        VortexSession::empty()
            .with::<DTypeSession>()
            .with::<ArraySession>()
            .with::<KernelSession>()
            .with::<AggregateFnSession>()
    });

    pub(crate) fn assert_io_error_kind(err: &VortexError, kind: io::ErrorKind) {
        let io_err = err
            .source()
            .and_then(|s| s.downcast_ref::<io::Error>())
            .vortex_expect("expected io::Error source");
        assert_eq!(io_err.kind(), kind);
    }

    pub(crate) struct FaultyReader {
        data: Vec<u8>,
        pos: usize,
        faults: Vec<(usize, io::ErrorKind)>,
        yield_pending: bool,
        calls: Arc<AtomicUsize>,
    }

    impl FaultyReader {
        pub(crate) fn new(data: Vec<u8>, faults: Vec<(usize, io::ErrorKind)>) -> Self {
            Self {
                data,
                pos: 0,
                faults,
                yield_pending: true,
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        pub(crate) fn calls(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.calls)
        }

        fn step(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if let Some(idx) = self.faults.iter().position(|(pos, _)| *pos == self.pos) {
                let (_, kind) = self.faults.remove(idx);
                return Err(io::Error::new(kind, "injected read error"));
            }
            if buf.is_empty() || self.pos >= self.data.len() {
                return Ok(0);
            }
            buf[0] = self.data[self.pos];
            self.pos += 1;
            Ok(1)
        }
    }

    impl io::Read for FaultyReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.step(buf)
        }
    }

    impl AsyncRead for FaultyReader {
        fn poll_read(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.yield_pending {
                self.yield_pending = false;
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
            self.yield_pending = true;
            Poll::Ready(self.step(buf))
        }
    }
}

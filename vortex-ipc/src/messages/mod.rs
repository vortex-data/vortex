// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod decoder;
mod encoder;
mod reader_async;
mod reader_buf;
mod reader_sync;
mod writer_async;
mod writer_sync;

pub use decoder::*;
pub use encoder::*;
pub use reader_async::*;
pub use reader_buf::*;
pub use reader_sync::*;
pub use writer_async::*;
pub use writer_sync::*;

#[cfg(test)]
mod tests {
    use std::io;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::task::Context;
    use std::task::Poll;

    use futures::AsyncRead;
    use futures::StreamExt;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_panic;

    use super::*;
    use crate::test::SESSION;
    use crate::test::assert_io_error_kind;

    struct FaultyReader {
        data: Vec<u8>,
        pos: usize,
        faults: Vec<(usize, io::ErrorKind)>,
        yield_pending: bool,
        calls: Arc<AtomicUsize>,
    }

    impl FaultyReader {
        fn new(data: Vec<u8>, faults: Vec<(usize, io::ErrorKind)>) -> Self {
            Self {
                data,
                pos: 0,
                faults,
                yield_pending: true,
                calls: Arc::new(AtomicUsize::new(0)),
            }
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

    struct AsyncAsIter {
        rt: tokio::runtime::Runtime,
        stream: AsyncMessageReader<FaultyReader>,
    }

    impl Iterator for AsyncAsIter {
        type Item = VortexResult<DecoderMessage>;

        fn next(&mut self) -> Option<Self::Item> {
            self.rt.block_on(self.stream.next())
        }
    }

    #[derive(Clone, Copy)]
    enum ReaderKind {
        Sync,
        Async,
    }

    impl ReaderKind {
        fn open(
            self,
            reader: FaultyReader,
        ) -> Box<dyn Iterator<Item = VortexResult<DecoderMessage>>> {
            match self {
                Self::Sync => Box::new(SyncMessageReader::new(reader)),
                Self::Async => Box::new(AsyncAsIter {
                    rt: tokio::runtime::Builder::new_current_thread()
                        .build()
                        .vortex_expect("runtime"),
                    stream: AsyncMessageReader::new(reader),
                }),
            }
        }
    }

    fn encoded_test_array() -> VortexResult<(ArrayRef, Vec<u8>, usize)> {
        let array = buffer![7i32, 8, 9].into_array();
        let mut encoder = MessageEncoder::new(SESSION.clone());
        let parts = encoder.encode(EncoderMessage::Array(&array))?;
        let header_len = parts[1].len();
        let bytes = parts.iter().flat_map(|b| b.as_ref()).copied().collect();
        Ok((array, bytes, header_len))
    }

    fn assert_decodes_to(
        msg: Option<VortexResult<DecoderMessage>>,
        expected: &ArrayRef,
    ) -> VortexResult<()> {
        let DecoderMessage::Array((parts, ctx, row_count)) =
            msg.vortex_expect("expected message")?
        else {
            vortex_panic!("expected an array message");
        };
        let actual = parts.decode(expected.dtype(), row_count, &ctx, &SESSION)?;
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[derive(Clone, Copy)]
    enum FaultOffset {
        Length,
        Header,
        Body,
    }

    impl FaultOffset {
        fn byte_pos(self, header_len: usize) -> usize {
            match self {
                Self::Length => 1,
                Self::Header => 4 + header_len / 2,
                Self::Body => 4 + header_len + 1,
            }
        }
    }

    #[rstest]
    fn resumable_io_error_preserves_progress(
        #[values(ReaderKind::Sync, ReaderKind::Async)] mode: ReaderKind,
        #[values(io::ErrorKind::WouldBlock, io::ErrorKind::TimedOut)] kind: io::ErrorKind,
        #[values(FaultOffset::Length, FaultOffset::Header, FaultOffset::Body)] offset: FaultOffset,
    ) -> VortexResult<()> {
        let (array, bytes, header_len) = encoded_test_array()?;
        let mut messages = mode.open(FaultyReader::new(
            bytes,
            vec![(offset.byte_pos(header_len), kind)],
        ));

        let err = messages
            .next()
            .vortex_expect("expected transport error")
            .unwrap_err();
        assert_io_error_kind(&err, kind);
        assert_decodes_to(messages.next(), &array)?;
        assert!(messages.next().is_none());
        Ok(())
    }

    #[rstest]
    fn interrupted_read_retries_and_preserves_progress(
        #[values(ReaderKind::Sync, ReaderKind::Async)] mode: ReaderKind,
    ) -> VortexResult<()> {
        let (array, bytes, header_len) = encoded_test_array()?;
        let mut messages = mode.open(FaultyReader::new(
            bytes,
            vec![
                (
                    FaultOffset::Length.byte_pos(header_len),
                    io::ErrorKind::Interrupted,
                ),
                (
                    FaultOffset::Header.byte_pos(header_len),
                    io::ErrorKind::Interrupted,
                ),
                (
                    FaultOffset::Body.byte_pos(header_len),
                    io::ErrorKind::Interrupted,
                ),
            ],
        ));

        assert_decodes_to(messages.next(), &array)?;
        assert!(messages.next().is_none());
        Ok(())
    }

    #[rstest]
    fn io_error_during_fill_is_terminal(
        #[values(ReaderKind::Sync, ReaderKind::Async)] mode: ReaderKind,
        #[values(FaultOffset::Length, FaultOffset::Header, FaultOffset::Body)] offset: FaultOffset,
    ) -> VortexResult<()> {
        let (_, bytes, header_len) = encoded_test_array()?;
        let reader = FaultyReader::new(
            bytes,
            vec![(offset.byte_pos(header_len), io::ErrorKind::BrokenPipe)],
        );
        let calls = Arc::clone(&reader.calls);
        let mut messages = mode.open(reader);

        let err = messages
            .next()
            .vortex_expect("expected io error")
            .unwrap_err();
        assert_io_error_kind(&err, io::ErrorKind::BrokenPipe);

        let calls_after_err = calls.load(Ordering::SeqCst);
        assert!(messages.next().is_none());
        assert!(messages.next().is_none());
        assert_eq!(calls.load(Ordering::SeqCst), calls_after_err);
        Ok(())
    }
}

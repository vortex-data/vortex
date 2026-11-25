// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;
use std::task::ready;

use bytes::BytesMut;
use futures::AsyncRead;
use futures::Stream;
use pin_project_lite::pin_project;
use vortex_error::VortexResult;

use crate::messages::DecoderMessage;
use crate::messages::MessageDecoder;
use crate::messages::PollRead;

pin_project! {
    /// An IPC message reader backed by an `AsyncRead` stream.
    pub struct AsyncMessageReader<R> {
        #[pin]
        read: R,
        buffer: BytesMut,
        decoder: MessageDecoder,
        state: ReadState,
    }
}

impl<R> AsyncMessageReader<R> {
    pub fn new(read: R) -> Self {
        AsyncMessageReader {
            read,
            buffer: BytesMut::new(),
            decoder: MessageDecoder::default(),
            state: ReadState::default(),
        }
    }
}

/// The state of an in-progress read operation.
#[derive(Default)]
enum ReadState {
    /// Ready to consult the decoder for the next operation.
    #[default]
    AwaitingDecoder,
    /// Filling the buffer with data from the underlying reader.
    ///
    /// Async readers may return fewer bytes than requested (partial reads), especially over network
    /// connections. This state persists across multiple `poll_next` calls until the buffer is
    /// completely filled, at which point we transition back to [`Self::AwaitingDecoder`].
    Filling {
        /// The number of bytes read into the buffer so far.
        total_bytes_read: usize,
    },
    /// Terminal state after clean EOF or a fatal I/O, decoder, or EOF error.
    Done,
}

/// Result of polling the reader to fill the buffer.
enum FillResult {
    /// The buffer has been completely filled.
    Filled,
    /// Need more data (partial read occurred).
    Pending,
    /// Underlying reader reached EOF (`poll_read` returned `0`).
    Eof,
}

/// Polls the reader to fill the buffer, handling partial reads.
fn poll_fill_buffer<R: AsyncRead>(
    read: Pin<&mut R>,
    buffer: &mut [u8],
    total_bytes_read: &mut usize,
    cx: &mut Context<'_>,
) -> Poll<io::Result<FillResult>> {
    let unfilled = &mut buffer[*total_bytes_read..];

    let bytes_read = ready!(read.poll_read(cx, unfilled))?;

    // `0` bytes read indicates an EOF.
    Poll::Ready(if bytes_read == 0 {
        Ok(FillResult::Eof)
    } else {
        *total_bytes_read += bytes_read;
        if *total_bytes_read == buffer.len() {
            Ok(FillResult::Filled)
        } else {
            debug_assert!(*total_bytes_read < buffer.len());
            Ok(FillResult::Pending)
        }
    })
}

impl<R: AsyncRead> Stream for AsyncMessageReader<R> {
    type Item = VortexResult<DecoderMessage>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        loop {
            match this.state {
                ReadState::Done => return Poll::Ready(None),
                ReadState::AwaitingDecoder => match this.decoder.read_next(this.buffer) {
                    Ok(PollRead::Some(msg)) => return Poll::Ready(Some(Ok(msg))),
                    Ok(PollRead::NeedMore(new_len)) => {
                        this.buffer.resize(new_len, 0x00);
                        *this.state = ReadState::Filling {
                            total_bytes_read: 0,
                        };
                    }
                    Err(e) => {
                        this.buffer.clear();
                        *this.state = ReadState::Done;
                        return Poll::Ready(Some(Err(e)));
                    }
                },
                ReadState::Filling { total_bytes_read } => {
                    match ready!(poll_fill_buffer(
                        this.read.as_mut(),
                        this.buffer,
                        total_bytes_read,
                        cx
                    )) {
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                        Err(e) => {
                            this.buffer.clear();
                            *this.state = ReadState::Done;
                            return Poll::Ready(Some(Err(e.into())));
                        }
                        Ok(FillResult::Eof) => {
                            let bytes_read = *total_bytes_read;
                            this.buffer.clear();
                            *this.state = ReadState::Done;
                            return match this.decoder.finish_eof(bytes_read) {
                                Ok(()) => Poll::Ready(None),
                                Err(e) => Poll::Ready(Some(Err(e))),
                            };
                        }
                        Ok(FillResult::Filled) => *this.state = ReadState::AwaitingDecoder,
                        Ok(FillResult::Pending) => {}
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::task::Context;
    use std::task::Poll;

    use bytes::BytesMut;
    use futures::AsyncRead;
    use futures::StreamExt;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_panic;

    use super::*;
    use crate::messages::EncoderMessage;
    use crate::messages::MessageEncoder;
    use crate::tests::SESSION;
    use crate::tests::assert_io_error_kind;

    /// An `AsyncRead` that alternates `Poll::Pending` with reads of up to `chunk_size` bytes.
    struct PendingChunkedReader {
        data: Vec<u8>,
        pos: usize,
        chunk_size: usize,
        yield_pending: bool,
    }

    impl PendingChunkedReader {
        fn new(data: Vec<u8>, chunk_size: usize) -> Self {
            Self {
                data,
                pos: 0,
                chunk_size,
                yield_pending: true,
            }
        }
    }

    impl AsyncRead for PendingChunkedReader {
        fn poll_read(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            if self.yield_pending {
                self.yield_pending = false;
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
            self.yield_pending = true;

            if buf.is_empty() || self.pos >= self.data.len() {
                return Poll::Ready(Ok(0));
            }
            let remaining = self.data.len() - self.pos;
            let count = buf.len().min(self.chunk_size).min(remaining);
            buf[..count].copy_from_slice(&self.data[self.pos..self.pos + count]);
            self.pos += count;
            Poll::Ready(Ok(count))
        }
    }

    #[tokio::test]
    async fn reads_messages_split_across_small_chunk_sizes() -> VortexResult<()> {
        let array = buffer![0i32, 1, 2, 3, 4, 5].into_array();
        let raw_buf = ByteBuffer::copy_from_aligned([10u8, 20, 30, 40, 50], Alignment::new(64));
        let mut ipc_bytes = BytesMut::new();
        let mut encoder = MessageEncoder::new(SESSION.clone());
        for message in [
            EncoderMessage::DType(array.dtype()),
            EncoderMessage::Array(&array),
            EncoderMessage::Buffer(&raw_buf),
        ] {
            for buf in encoder.encode(message)? {
                ipc_bytes.extend_from_slice(buf.as_ref());
            }
        }
        let encoded = ipc_bytes.to_vec();

        for chunk_size in [1, 2, 3, 7, 13] {
            let mut reader =
                AsyncMessageReader::new(PendingChunkedReader::new(encoded.clone(), chunk_size));

            let DecoderMessage::DType(fb_dtype) =
                reader.next().await.vortex_expect("dtype message")?
            else {
                vortex_panic!("expected a dtype message");
            };
            assert_eq!(DType::from_flatbuffer(fb_dtype, &SESSION)?, *array.dtype());

            let DecoderMessage::Array((parts, ctx, row_count)) =
                reader.next().await.vortex_expect("array message")?
            else {
                vortex_panic!("expected an array message");
            };
            let actual = parts.decode(array.dtype(), row_count, &ctx, &SESSION)?;
            assert_arrays_eq!(actual, array, &mut SESSION.create_execution_ctx());

            let DecoderMessage::Buffer(decoded_buf) =
                reader.next().await.vortex_expect("buffer message")?
            else {
                vortex_panic!("expected a buffer message");
            };
            assert_eq!(decoded_buf.as_slice(), raw_buf.as_slice());
            assert_eq!(decoded_buf.alignment(), Alignment::new(64));
            assert!(Alignment::new(64).is_ptr_aligned(decoded_buf.as_ptr()));

            assert!(reader.next().await.is_none());
        }

        Ok(())
    }

    #[tokio::test]
    async fn truncated_message_errors_once_then_terminates() -> VortexResult<()> {
        let array = buffer![0i32, 1, 2, 3].into_array();
        let mut encoder = MessageEncoder::new(SESSION.clone());
        let parts = encoder.encode(EncoderMessage::Array(&array))?;
        let header_len = parts[1].len();
        let body_len: usize = parts[2..].iter().map(|b| b.len()).sum();
        let full: Vec<u8> = parts.iter().flat_map(|b| b.as_ref()).copied().collect();

        let cut_points = [
            2,
            4,
            4 + header_len / 2,
            4 + header_len,
            4 + header_len + body_len / 2,
        ];

        for cut in cut_points {
            let mut reader =
                AsyncMessageReader::new(PendingChunkedReader::new(full[..cut].to_vec(), 1));
            let Err(err) = reader.next().await.vortex_expect("truncated frame") else {
                vortex_panic!("expected error on truncated frame");
            };
            assert_io_error_kind(&err, io::ErrorKind::UnexpectedEof);
            assert!(reader.next().await.is_none());
            assert!(reader.next().await.is_none());
        }

        Ok(())
    }

    #[tokio::test]
    async fn interrupted_read_retries_and_preserves_progress() -> VortexResult<()> {
        struct InterruptedAsyncReader {
            data: Vec<u8>,
            pos: usize,
            interrupt_positions: Vec<usize>,
            yield_pending: bool,
        }

        impl AsyncRead for InterruptedAsyncReader {
            fn poll_read(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                buf: &mut [u8],
            ) -> Poll<io::Result<usize>> {
                if self.yield_pending {
                    self.yield_pending = false;
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                self.yield_pending = true;

                if let Some(idx) = self.interrupt_positions.iter().position(|p| *p == self.pos) {
                    self.interrupt_positions.remove(idx);
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "transient interrupt",
                    )));
                }
                if buf.is_empty() || self.pos >= self.data.len() {
                    return Poll::Ready(Ok(0));
                }
                buf[0] = self.data[self.pos];
                self.pos += 1;
                Poll::Ready(Ok(1))
            }
        }

        let array = buffer![7i32, 8, 9].into_array();
        let mut encoder = MessageEncoder::new(SESSION.clone());
        let parts = encoder.encode(EncoderMessage::Array(&array))?;
        let header_len = parts[1].len();
        let full: Vec<u8> = parts.iter().flat_map(|b| b.as_ref()).copied().collect();

        let mut reader = AsyncMessageReader::new(InterruptedAsyncReader {
            data: full,
            pos: 0,
            interrupt_positions: vec![1, 4 + header_len / 2, 4 + header_len + 1],
            yield_pending: true,
        });

        let DecoderMessage::Array((parts, ctx, row_count)) =
            reader.next().await.vortex_expect("recovered message")?
        else {
            vortex_panic!("expected an array message");
        };
        let actual = parts.decode(array.dtype(), row_count, &ctx, &SESSION)?;
        assert_arrays_eq!(actual, array, &mut SESSION.create_execution_ctx());
        assert!(reader.next().await.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn io_error_during_fill_is_terminal() -> VortexResult<()> {
        struct FailingAsyncReader {
            data: Vec<u8>,
            pos: usize,
            fail_at: usize,
            yield_pending: bool,
            poll_calls: Arc<AtomicUsize>,
        }

        impl AsyncRead for FailingAsyncReader {
            fn poll_read(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                buf: &mut [u8],
            ) -> Poll<io::Result<usize>> {
                self.poll_calls.fetch_add(1, Ordering::SeqCst);
                if self.yield_pending {
                    self.yield_pending = false;
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                self.yield_pending = true;

                if self.pos >= self.fail_at {
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "broken transport",
                    )));
                }
                if buf.is_empty() || self.pos >= self.data.len() {
                    return Poll::Ready(Ok(0));
                }
                buf[0] = self.data[self.pos];
                self.pos += 1;
                Poll::Ready(Ok(1))
            }
        }

        let array = buffer![7i32, 8, 9].into_array();
        let mut encoder = MessageEncoder::new(SESSION.clone());
        let parts = encoder.encode(EncoderMessage::Array(&array))?;
        let header_len = parts[1].len();
        let full: Vec<u8> = parts.iter().flat_map(|b| b.as_ref()).copied().collect();

        for fail_at in [1, 4 + header_len / 2, 4 + header_len + 1] {
            let poll_calls = Arc::new(AtomicUsize::new(0));
            let mut reader = AsyncMessageReader::new(FailingAsyncReader {
                data: full.clone(),
                pos: 0,
                fail_at,
                yield_pending: true,
                poll_calls: Arc::clone(&poll_calls),
            });

            let Err(err) = reader.next().await.vortex_expect("expected io error") else {
                vortex_panic!("expected io error");
            };
            assert_io_error_kind(&err, io::ErrorKind::BrokenPipe);

            let calls_after_err = poll_calls.load(Ordering::SeqCst);
            assert!(reader.next().await.is_none());
            assert!(reader.next().await.is_none());
            assert_eq!(poll_calls.load(Ordering::SeqCst), calls_after_err);
        }

        Ok(())
    }
}

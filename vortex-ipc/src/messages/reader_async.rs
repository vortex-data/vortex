// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
    /// Terminal state after clean EOF or a fatal decoder/EOF error.
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
) -> Poll<VortexResult<FillResult>> {
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
                        Err(e) => return Poll::Ready(Some(Err(e))),
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
    use std::task::Context;
    use std::task::Poll;

    use futures::AsyncRead;
    use futures::StreamExt;
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use super::*;
    use crate::messages::EncoderMessage;
    use crate::messages::MessageEncoder;
    use crate::test::SESSION;
    use crate::test::assert_io_error_kind;

    /// An `AsyncRead` that alternates `Poll::Pending` with 1-byte reads.
    struct PendingByteReader {
        data: Vec<u8>,
        pos: usize,
        yield_pending: bool,
    }

    impl PendingByteReader {
        fn new(data: Vec<u8>) -> Self {
            Self {
                data,
                pos: 0,
                yield_pending: true,
            }
        }
    }

    impl AsyncRead for PendingByteReader {
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
            buf[0] = self.data[self.pos];
            self.pos += 1;
            Poll::Ready(Ok(1))
        }
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
            let mut reader = AsyncMessageReader::new(PendingByteReader::new(full[..cut].to_vec()));
            let err = reader
                .next()
                .await
                .vortex_expect("expected error on truncated frame")
                .unwrap_err();
            assert_io_error_kind(&err, io::ErrorKind::UnexpectedEof);
            assert!(reader.next().await.is_none());
            assert!(reader.next().await.is_none());
        }

        Ok(())
    }
}

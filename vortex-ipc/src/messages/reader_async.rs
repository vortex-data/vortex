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
use vortex_array::session::ArrayRegistry;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

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
    pub fn new(read: R, registry: ArrayRegistry) -> Self {
        AsyncMessageReader {
            read,
            buffer: BytesMut::new(),
            decoder: MessageDecoder::new(registry),
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
}

/// Result of polling the reader to fill the buffer.
enum FillResult {
    /// The buffer has been completely filled.
    Filled,
    /// Need more data (partial read occurred).
    Pending,
    /// Clean EOF at a message boundary.
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
        if *total_bytes_read > 0 {
            Err(vortex_err!(
                "unexpected EOF during partial read: read {total_bytes_read} of {} expected bytes",
                buffer.len()
            ))
        } else {
            Ok(FillResult::Eof)
        }
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
                ReadState::AwaitingDecoder => match this.decoder.read_next(this.buffer)? {
                    PollRead::Some(msg) => return Poll::Ready(Some(Ok(msg))),
                    PollRead::NeedMore(new_len) => {
                        this.buffer.resize(new_len, 0x00);
                        *this.state = ReadState::Filling {
                            total_bytes_read: 0,
                        };
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
                        Ok(FillResult::Eof) => return Poll::Ready(None),
                        Ok(FillResult::Filled) => *this.state = ReadState::AwaitingDecoder,
                        Ok(FillResult::Pending) => {}
                    }
                }
            }
        }
    }
}

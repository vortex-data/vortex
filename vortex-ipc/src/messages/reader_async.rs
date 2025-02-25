use std::pin::Pin;
use std::task::{Context, Poll, ready};

use bytes::BytesMut;
use futures_util::{AsyncRead, Stream};
use pin_project_lite::pin_project;
use vortex_error::VortexResult;

use crate::messages::{DecoderMessage, MessageDecoder, PollRead};

pin_project! {
    /// An IPC message reader backed by an `AsyncRead` stream.
    pub struct AsyncMessageReader<R> {
        #[pin]
        read: R,
        buffer: BytesMut,
        decoder: MessageDecoder,
        bytes_read: usize,
    }
}

impl<R> AsyncMessageReader<R> {
    pub fn new(read: R) -> Self {
        AsyncMessageReader {
            read,
            buffer: BytesMut::new(),
            decoder: MessageDecoder::default(),
            bytes_read: 0,
        }
    }
}

impl<R: AsyncRead> Stream for AsyncMessageReader<R> {
    type Item = VortexResult<DecoderMessage>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        loop {
            match this.decoder.read_next(this.buffer)? {
                PollRead::Some(msg) => return Poll::Ready(Some(Ok(msg))),
                PollRead::NeedMore(nbytes) => {
                    this.buffer.resize(nbytes, 0x00);

                    match ready!(
                        this.read
                            .as_mut()
                            .poll_read(cx, &mut this.buffer.as_mut()[*this.bytes_read..])
                    ) {
                        Ok(0) => {
                            // End of file
                            return Poll::Ready(None);
                        }
                        Ok(nbytes) => {
                            *this.bytes_read += nbytes;
                            // If we've finished the read operation, then we continue the loop
                            // and the decoder should present us with a new response.
                            if *this.bytes_read == nbytes {
                                *this.bytes_read = 0;
                            }
                        }
                        Err(e) => return Poll::Ready(Some(Err(e.into()))),
                    }
                }
            }
        }
    }
}

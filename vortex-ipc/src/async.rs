use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Poll};

use bytes::BytesMut;
use futures_util::{AsyncRead, AsyncReadExt, FutureExt, Stream, StreamExt};
use vortex_array::stream::ArrayStream;
use vortex_array::{ArrayDType, ArrayData, Context};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::messages::{
    DecoderMessage, EncoderMessage, MessageDecoder, MessageEncoder, NextMessage,
};
use crate::ALIGNMENT;

/// An [`ArrayStream`] for reading messages off an async IPC stream.
pub struct IPCArrayStream<R: AsyncRead> {
    read: R,
    ctx: Arc<Context>,
    dtype: DType,
    buffer: BytesMut,
    decoder: MessageDecoder,
}

impl<R: AsyncRead + Unpin> IPCArrayStream<R> {
    pub async fn try_new(mut read: R, ctx: Arc<Context>) -> VortexResult<Self> {
        let mut message_reader = MessageDecoder::default();
        let mut buffer = BytesMut::new();

        loop {
            match message_reader.read_next(&mut buffer)? {
                NextMessage::Some(msg) => {
                    if let DecoderMessage::DType(dtype) = msg {
                        return Ok(IPCArrayStream {
                            read,
                            ctx,
                            dtype,
                            buffer,
                            decoder: message_reader,
                        });
                    } else {
                        vortex_bail!("Expected DType message, got {:?}", msg);
                    }
                }
                NextMessage::NeedMore(nbytes) => {
                    buffer.resize(nbytes, 0x00);
                    read.read_exact(&mut buffer).await.map_err(|e| {
                        VortexError::Context(
                            "IO error reading initial DType from stream".into(),
                            Box::new(e.into()),
                        )
                    })?;
                }
            }
        }
    }
}

impl<R: AsyncRead + Unpin> ArrayStream for IPCArrayStream<R> {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl<R: AsyncRead + Unpin> Stream for IPCArrayStream<R> {
    type Item = VortexResult<ArrayData>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match this.decoder.read_next(&mut this.buffer) {
                Ok(NextMessage::Some(DecoderMessage::Array(components))) => {
                    return Poll::Ready(Some(
                        components
                            .into_array_data(this.ctx.clone(), this.dtype.clone())
                            .and_then(|array| {
                                if array.dtype() != this.dtype() {
                                    Err(vortex_err!(
                                        "Array data type mismatch: expected {:?}, got {:?}",
                                        this.dtype(),
                                        array.dtype()
                                    ))
                                } else {
                                    Ok(array)
                                }
                            }),
                    ));
                }
                Ok(NextMessage::Some(DecoderMessage::Buffer(_))) => {
                    return Poll::Ready(Some(Err(vortex_err!(
                        "Unexpected Buffer message in IPC stream"
                    ))));
                }
                Ok(NextMessage::Some(DecoderMessage::DType(_))) => {
                    return Poll::Ready(Some(Err(vortex_err!(
                        "Unexpected DType message in IPC stream"
                    ))));
                }
                Ok(NextMessage::NeedMore(nbytes)) => {
                    this.buffer.resize(nbytes, 0x00);
                    match ready!(this.read.read(&mut this.buffer).poll_unpin(cx)) {
                        Ok(0) => {
                            // Reached EOF
                            return Poll::Ready(None);
                        }
                        Ok(_nbytes) => {
                            // Continue the loop to try reading the next IPC message
                        }
                        Err(e) => return Poll::Ready(Some(Err(e.into()))),
                    }
                }
                Err(e) => return Poll::Ready(Some(Err(e.into()))),
            }
        }
    }
}

/// A trait for convering an [`ArrayStream`] into IPC streams.
pub trait ArrayStreamIntoIPC {
    fn into_ipc_bytes(self) -> ArrayStreamIntoIPCBytes
    where
        Self: Sized;
}

impl<S: ArrayStream + 'static> ArrayStreamIntoIPC for S {
    fn into_ipc_bytes(self) -> ArrayStreamIntoIPCBytes
    where
        Self: Sized,
    {
        ArrayStreamIntoIPCBytes {
            stream: Box::pin(self),
            encoder: MessageEncoder::new(ALIGNMENT as u16),
            buffers: vec![],
            written_dtype: false,
        }
    }
}

pub struct ArrayStreamIntoIPCBytes {
    stream: Pin<Box<dyn ArrayStream + 'static>>,
    encoder: MessageEncoder,
    buffers: Vec<Buffer>,
    written_dtype: bool,
}

impl ArrayStreamIntoIPCBytes {
    /// Collects the IPC bytes into a single Vec<u8> buffer.
    pub async fn collect_to_buffer(mut self) -> VortexResult<Vec<u8>> {
        let mut buffer = vec![];
        while let Some(chunk) = self.next().await {
            buffer.extend(chunk?.as_slice());
        }
        Ok(buffer)
    }
}

impl Stream for ArrayStreamIntoIPCBytes {
    type Item = VortexResult<Buffer>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // If we haven't written the dtype yet, we write it
        if !this.written_dtype {
            this.buffers.extend(
                this.encoder
                    .encode(EncoderMessage::DType(this.stream.dtype())),
            );
            this.written_dtype = true;
        }

        // Try to flush any buffers we have
        if !this.buffers.is_empty() {
            return Poll::Ready(Some(Ok(this.buffers.remove(0))));
        }

        // Or else try to serialize the next array
        match ready!(this.stream.poll_next_unpin(cx)) {
            None => return Poll::Ready(None),
            Some(chunk) => match chunk {
                Ok(chunk) => {
                    this.buffers
                        .extend(this.encoder.encode(EncoderMessage::Array(&chunk)));
                }
                Err(e) => return Poll::Ready(Some(Err(e))),
            },
        }

        // Try to flush any buffers we have again
        if !this.buffers.is_empty() {
            return Poll::Ready(Some(Ok(this.buffers.remove(0))));
        }

        // Otherwise, we're done
        Poll::Ready(None)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use futures_util::io::Cursor;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::stream::{ArrayStream, ArrayStreamExt};
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayDType, Context, IntoArrayVariant, ToArrayData};

    use super::*;

    #[tokio::test]
    async fn test_async_stream() {
        let array = PrimitiveArray::from_vec::<i32>(vec![1, 2, 3], Validity::NonNullable);
        let ipc_buffer = array
            .to_array()
            .into_array_stream()
            .into_ipc_bytes()
            .collect_to_buffer()
            .await
            .unwrap();

        let reader = IPCArrayStream::try_new(Cursor::new(ipc_buffer), Arc::new(Context::default()))
            .await
            .unwrap();

        assert_eq!(reader.dtype(), array.dtype());
        let result = reader
            .into_array_data()
            .await
            .unwrap()
            .into_primitive()
            .unwrap();
        println!("{:?}", result);
        assert_eq!(
            array.maybe_null_slice::<i32>(),
            result.maybe_null_slice::<i32>()
        );
    }
}

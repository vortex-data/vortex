use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Poll};

use aligned_buffer::UniqueAlignedBuffer;
use bytes::Bytes;
use futures_util::{AsyncRead, AsyncWrite, AsyncWriteExt, Stream, StreamExt, TryStreamExt};
use pin_project_lite::pin_project;
use vortex_array::stream::ArrayStream;
use vortex_array::{ArrayDType, ArrayData, Context};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::messages::{AsyncMessageReader, DecoderMessage, EncoderMessage, MessageEncoder};
use crate::ALIGNMENT;

pin_project! {
    /// An [`ArrayStream`] for reading messages off an async IPC stream.
    pub struct AsyncIPCReader<R> {
        #[pin]
        reader: AsyncMessageReader<R>,
        ctx: Arc<Context>,
        dtype: DType,
    }
}

impl<R: AsyncRead + Unpin> AsyncIPCReader<R> {
    pub async fn try_new(read: R, ctx: Arc<Context>) -> VortexResult<Self> {
        let mut reader = AsyncMessageReader::new(read);

        let dtype = match reader.next().await.transpose()? {
            Some(msg) => match msg {
                DecoderMessage::DType(dtype) => dtype,
                msg => {
                    vortex_bail!("Expected DType message, got {:?}", msg);
                }
            },
            None => vortex_bail!("Expected DType message, got EOF"),
        };

        Ok(AsyncIPCReader { reader, ctx, dtype })
    }
}

impl<R: AsyncRead> ArrayStream for AsyncIPCReader<R> {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl<R: AsyncRead> Stream for AsyncIPCReader<R> {
    type Item = VortexResult<ArrayData>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.project();

        match ready!(this.reader.poll_next(cx)) {
            None => Poll::Ready(None),
            Some(msg) => match msg {
                Ok(DecoderMessage::Array(array_parts)) => Poll::Ready(Some(
                    array_parts
                        .into_array_data(this.ctx.clone(), this.dtype.clone())
                        .and_then(|array| {
                            if array.dtype() != this.dtype {
                                Err(vortex_err!(
                                    "Array data type mismatch: expected {:?}, got {:?}",
                                    this.dtype,
                                    array.dtype()
                                ))
                            } else {
                                Ok(array)
                            }
                        }),
                )),
                Ok(msg) => Poll::Ready(Some(Err(vortex_err!(
                    "Expected Array message, got {:?}",
                    msg
                )))),
                Err(e) => Poll::Ready(Some(Err(e))),
            },
        }
    }
}

/// A trait for convering an [`ArrayStream`] into IPC streams.
pub trait ArrayStreamIPC {
    fn into_ipc(self) -> ArrayStreamIPCBytes
    where
        Self: Sized;

    fn write_ipc<W: AsyncWrite + Unpin>(self, write: W) -> impl Future<Output = VortexResult<W>>
    where
        Self: Sized;
}

impl<S: ArrayStream + 'static> ArrayStreamIPC for S {
    fn into_ipc(self) -> ArrayStreamIPCBytes
    where
        Self: Sized,
    {
        ArrayStreamIPCBytes {
            stream: Box::pin(self),
            encoder: MessageEncoder::default(),
            buffers: vec![],
            written_dtype: false,
        }
    }

    async fn write_ipc<W: AsyncWrite + Unpin>(self, mut write: W) -> VortexResult<W>
    where
        Self: Sized,
    {
        let mut stream = self.into_ipc();
        while let Some(chunk) = stream.next().await {
            write.write_all(chunk?.as_slice()).await?;
        }
        Ok(write)
    }
}

pub struct ArrayStreamIPCBytes {
    stream: Pin<Box<dyn ArrayStream + 'static>>,
    encoder: MessageEncoder,
    buffers: Vec<Buffer>,
    written_dtype: bool,
}

impl ArrayStreamIPCBytes {
    /// Collects the IPC bytes into a single `Buffer`.
    pub async fn collect_to_buffer(self) -> VortexResult<Buffer> {
        // We allocate a single aligned buffer to hold the combined IPC bytes
        let buffers: Vec<Buffer> = self.try_collect().await?;
        let mut buffer =
            UniqueAlignedBuffer::<ALIGNMENT>::with_capacity(buffers.iter().map(|b| b.len()).sum());
        for buf in buffers {
            buffer.extend_from_slice(buf.as_slice());
        }
        Ok(Buffer::from(Bytes::from_owner(buffer)))
    }
}

impl Stream for ArrayStreamIPCBytes {
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
            .into_ipc()
            .collect_to_buffer()
            .await
            .unwrap();

        let reader = AsyncIPCReader::try_new(Cursor::new(ipc_buffer), Arc::new(Context::default()))
            .await
            .unwrap();

        assert_eq!(reader.dtype(), array.dtype());
        let result = reader
            .into_array_data()
            .await
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(
            array.maybe_null_slice::<i32>(),
            result.maybe_null_slice::<i32>()
        );
    }
}

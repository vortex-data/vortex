// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::pin::Pin;
use std::task::Poll;
use std::task::ready;

use bytes::Bytes;
use bytes::BytesMut;
use futures::AsyncRead;
use futures::AsyncWrite;
use futures::AsyncWriteExt;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use pin_project_lite::pin_project;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::messages::AsyncMessageReader;
use crate::messages::DecoderMessage;
use crate::messages::EncoderMessage;
use crate::messages::MessageEncoder;

pin_project! {
    /// An [`ArrayStream`] for reading messages off an async IPC stream.
    pub struct AsyncIPCReader<R> {
        #[pin]
        reader: AsyncMessageReader<R>,
        dtype: DType,
        session: VortexSession,
        done: bool,
    }
}

impl<R: AsyncRead + Unpin> AsyncIPCReader<R> {
    pub async fn try_new(read: R, session: &VortexSession) -> VortexResult<Self> {
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

        let dtype = DType::from_flatbuffer(dtype, session)?;

        Ok(AsyncIPCReader {
            reader,
            dtype,
            session: session.clone(),
            done: false,
        })
    }
}

impl<R: AsyncRead> ArrayStream for AsyncIPCReader<R> {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl<R: AsyncRead> Stream for AsyncIPCReader<R> {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.project();
        if *this.done {
            return Poll::Ready(None);
        }

        let Some(msg) = ready!(this.reader.poll_next(cx)) else {
            *this.done = true;
            return Poll::Ready(None);
        };
        let result = match msg {
            Ok(DecoderMessage::Array((array_parts, ctx, row_count))) => array_parts
                .decode(this.dtype, row_count, &ctx, this.session)
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
            Ok(msg) => Err(vortex_err!("Expected Array message, got {:?}", msg)),
            Err(e) => Err(e),
        };
        if result.is_err() {
            *this.done = true;
        }
        Poll::Ready(Some(result))
    }
}

/// A trait for converting an [`ArrayStream`] into IPC streams.
pub trait ArrayStreamIPC {
    fn into_ipc(self, session: &VortexSession) -> ArrayStreamIPCBytes
    where
        Self: Sized;

    fn write_ipc<W: AsyncWrite + Unpin>(
        self,
        write: W,
        session: &VortexSession,
    ) -> impl Future<Output = VortexResult<W>>
    where
        Self: Sized;
}

impl<S: ArrayStream + 'static> ArrayStreamIPC for S {
    fn into_ipc(self, session: &VortexSession) -> ArrayStreamIPCBytes
    where
        Self: Sized,
    {
        ArrayStreamIPCBytes {
            stream: Box::pin(self),
            encoder: MessageEncoder::new(session.clone()),
            buffers: vec![],
            written_dtype: false,
            done: false,
        }
    }

    async fn write_ipc<W: AsyncWrite + Unpin>(
        self,
        mut write: W,
        session: &VortexSession,
    ) -> VortexResult<W>
    where
        Self: Sized,
    {
        let mut stream = self.into_ipc(session);
        while let Some(chunk) = stream.next().await {
            write.write_all(&chunk?).await?;
        }
        Ok(write)
    }
}

pub struct ArrayStreamIPCBytes {
    stream: Pin<Box<dyn ArrayStream + 'static>>,
    encoder: MessageEncoder,
    buffers: Vec<Bytes>,
    written_dtype: bool,
    done: bool,
}

impl ArrayStreamIPCBytes {
    /// Collects the IPC bytes into a single `Bytes`.
    pub async fn collect_to_buffer(self) -> VortexResult<Bytes> {
        let buffers: Vec<Bytes> = self.try_collect().await?;
        let mut buffer = BytesMut::with_capacity(buffers.iter().map(|b| b.len()).sum());
        for buf in buffers {
            buffer.extend_from_slice(buf.as_ref());
        }
        Ok(buffer.freeze())
    }
}

impl Stream for ArrayStreamIPCBytes {
    type Item = VortexResult<Bytes>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.done {
            return Poll::Ready(None);
        }

        // If we haven't written the dtype yet, we write it
        if !this.written_dtype {
            let dtype = this.stream.dtype();
            match this.encoder.encode(EncoderMessage::DType(dtype)) {
                Ok(buffers) => {
                    this.buffers.extend(buffers);
                    this.written_dtype = true;
                }
                Err(e) => {
                    this.done = true;
                    let err = e.with_context("Failed to encode DType message");
                    return Poll::Ready(Some(Err(err)));
                }
            }
        }

        // Try to flush any buffers we have
        if !this.buffers.is_empty() {
            return Poll::Ready(Some(Ok(this.buffers.remove(0))));
        }

        // Or else try to serialize the next array
        match ready!(this.stream.poll_next_unpin(cx)) {
            None => {
                this.done = true;
                return Poll::Ready(None);
            }
            Some(chunk) => match chunk.and_then(|c| this.encoder.encode(EncoderMessage::Array(&c)))
            {
                Ok(buffers) => {
                    this.buffers.extend(buffers);
                }
                Err(e) => {
                    this.done = true;
                    return Poll::Ready(Some(Err(e)));
                }
            },
        }

        // Try to flush any buffers we have again
        if !this.buffers.is_empty() {
            return Poll::Ready(Some(Ok(this.buffers.remove(0))));
        }

        // Otherwise, we're done
        this.done = true;
        Poll::Ready(None)
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
    use std::vec::IntoIter;

    use futures::io::Cursor;
    use futures::stream;
    use vortex_array::IntoArray as _;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::stream::ArrayStreamAdapter;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::vortex_panic;

    use super::*;
    use crate::iterator::SyncIPCReader;
    use crate::messages::AsyncMessageWriter;
    use crate::tests::SESSION;
    use crate::tests::assert_io_error_kind;

    struct CountingIter {
        items: IntoIter<VortexResult<ArrayRef>>,
        pulled: Arc<AtomicUsize>,
    }

    impl Iterator for CountingIter {
        type Item = VortexResult<ArrayRef>;

        fn next(&mut self) -> Option<Self::Item> {
            self.pulled.fetch_add(1, Ordering::SeqCst);
            self.items.next()
        }
    }

    #[tokio::test]
    async fn test_async_stream() -> VortexResult<()> {
        let array = buffer![1, 2, 3].into_array();
        let ipc_buffer = array
            .to_array_stream()
            .into_ipc(&SESSION)
            .collect_to_buffer()
            .await?;

        let reader = AsyncIPCReader::try_new(Cursor::new(ipc_buffer), &SESSION).await?;

        assert_eq!(reader.dtype(), array.dtype());
        let result = reader.read_all().await?;
        assert_arrays_eq!(result, array, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[tokio::test]
    async fn test_async_write_ipc_roundtrip() -> VortexResult<()> {
        let chunk1 = buffer![1i32, 2, 3].into_array();
        let chunk2 = buffer![4i32, 5, 6, 7].into_array();
        let dtype = chunk1.dtype().clone();
        let expected = buffer![1i32, 2, 3, 4, 5, 6, 7].into_array();

        let cursor = ArrayStreamAdapter::new(dtype, stream::iter([Ok(chunk1), Ok(chunk2)]))
            .write_ipc(Cursor::new(Vec::new()), &SESSION)
            .await?;
        let reader = AsyncIPCReader::try_new(Cursor::new(cursor.into_inner()), &SESSION).await?;

        assert_eq!(reader.dtype(), expected.dtype());
        let result = reader.read_all().await?;
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    /// Wrapper that limits reads to small chunks to simulate network behavior
    struct ChunkedReader<R> {
        inner: R,
        chunk_size: usize,
    }

    impl<R: AsyncRead + Unpin> AsyncRead for ChunkedReader<R> {
        fn poll_read(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            let chunk_size = self.chunk_size.min(buf.len());
            Pin::new(&mut self.inner).poll_read(cx, &mut buf[..chunk_size])
        }
    }

    #[tokio::test]
    async fn test_async_stream_chunked() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array();
        let ipc_buffer = array
            .to_array_stream()
            .into_ipc(&SESSION)
            .collect_to_buffer()
            .await?;

        let chunked = ChunkedReader {
            inner: Cursor::new(ipc_buffer),
            chunk_size: 3,
        };

        let reader = AsyncIPCReader::try_new(chunked, &SESSION).await?;

        let result = reader.read_all().await?;
        let expected = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    /// Test with 1-byte chunks to stress-test partial read handling.
    #[tokio::test]
    async fn test_async_stream_single_byte_chunks() -> VortexResult<()> {
        let array = buffer![42i64, -1, 0, i64::MAX, i64::MIN].into_array();
        let ipc_buffer = array
            .to_array_stream()
            .into_ipc(&SESSION)
            .collect_to_buffer()
            .await?;

        let chunked = ChunkedReader {
            inner: Cursor::new(ipc_buffer),
            chunk_size: 1,
        };

        let reader = AsyncIPCReader::try_new(chunked, &SESSION).await?;

        let result = reader.read_all().await?;
        let expected = buffer![42i64, -1, 0, i64::MAX, i64::MIN].into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[tokio::test]
    async fn sync_and_async_adapters_reject_truncated_second_chunk() -> VortexResult<()> {
        let chunk1 = buffer![1i32, 2, 3].into_array();
        let chunk2 = buffer![4i32, 5, 6, 7].into_array();

        let mut encoder = MessageEncoder::new(SESSION.clone());
        let mut truncated = Vec::new();
        for part in encoder.encode(EncoderMessage::DType(chunk1.dtype()))? {
            truncated.extend_from_slice(part.as_ref());
        }
        for part in encoder.encode(EncoderMessage::Array(&chunk1))? {
            truncated.extend_from_slice(part.as_ref());
        }
        let chunk2_parts = encoder.encode(EncoderMessage::Array(&chunk2))?;
        // Include only the 4-byte length prefix of chunk 2.
        truncated.extend_from_slice(chunk2_parts[0].as_ref());

        let mut ctx = SESSION.create_execution_ctx();

        let mut sync_reader = SyncIPCReader::try_new(io::Cursor::new(truncated.clone()), &SESSION)?;
        let first = sync_reader.next().vortex_expect("expected first chunk")?;
        assert_arrays_eq!(first, chunk1, &mut ctx);
        let Err(err) = sync_reader.next().vortex_expect("truncated chunk") else {
            vortex_panic!("expected error on truncated chunk");
        };
        assert_io_error_kind(&err, io::ErrorKind::UnexpectedEof);
        assert!(sync_reader.next().is_none());

        let mut async_reader = AsyncIPCReader::try_new(Cursor::new(truncated), &SESSION).await?;
        let first = async_reader.next().await.vortex_expect("first chunk")?;
        assert_arrays_eq!(first, chunk1, &mut ctx);
        let Err(err) = async_reader.next().await.vortex_expect("truncated chunk") else {
            vortex_panic!("expected error on truncated chunk");
        };
        assert_io_error_kind(&err, io::ErrorKind::UnexpectedEof);
        assert!(async_reader.next().await.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_async_ipc_bytes_mid_stream_error_terminates() -> VortexResult<()> {
        let chunk1 = buffer![1i32, 2].into_array();
        let chunk2 = buffer![3i32, 4].into_array();
        let dtype = chunk1.dtype().clone();
        let pulled = Arc::new(AtomicUsize::new(0));
        let items = vec![
            Ok(chunk1.clone()),
            Err(vortex_err!("boom")),
            Ok(chunk2.clone()),
        ];

        let input = ArrayStreamAdapter::new(
            dtype.clone(),
            stream::iter(CountingIter {
                items: items.into_iter(),
                pulled: Arc::clone(&pulled),
            }),
        );
        let mut ipc_bytes = input.into_ipc(&SESSION);

        let mut seen_bytes = 0usize;
        let err = loop {
            match ipc_bytes.next().await.vortex_expect("mid-stream error") {
                Ok(buf) => seen_bytes += buf.len(),
                Err(err) => break err,
            }
        };
        assert!(seen_bytes > 0);
        assert!(err.to_string().contains("boom"));
        assert_eq!(pulled.load(Ordering::SeqCst), 2);
        assert!(ipc_bytes.next().await.is_none());
        assert_eq!(pulled.load(Ordering::SeqCst), 2);

        let err_stream = ArrayStreamAdapter::new(
            dtype,
            stream::iter([Ok(chunk1), Err(vortex_err!("boom")), Ok(chunk2)]),
        );
        let result = err_stream.into_ipc(&SESSION).collect_to_buffer().await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_async_ipc_reader_mid_stream_io_error_terminates() -> VortexResult<()> {
        struct FailAfterAsyncReader {
            data: Vec<u8>,
            pos: usize,
            fail_at: usize,
            poll_calls: Arc<AtomicUsize>,
        }

        impl AsyncRead for FailAfterAsyncReader {
            fn poll_read(
                mut self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
                buf: &mut [u8],
            ) -> Poll<io::Result<usize>> {
                self.poll_calls.fetch_add(1, Ordering::SeqCst);
                if self.pos >= self.fail_at {
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "mid-stream failure",
                    )));
                }
                let remaining = self.fail_at - self.pos;
                let count = buf.len().min(remaining);
                buf[..count].copy_from_slice(&self.data[self.pos..self.pos + count]);
                self.pos += count;
                Poll::Ready(Ok(count))
            }
        }

        let chunk1 = buffer![1i32, 2, 3].into_array();
        let chunk2 = buffer![4i32, 5, 6].into_array();
        let dtype_msg = EncoderMessage::DType(chunk1.dtype());
        let mut writer = AsyncMessageWriter::new(Cursor::new(Vec::new()), &SESSION);
        writer.write_message(dtype_msg).await?;
        writer.write_message(EncoderMessage::Array(&chunk1)).await?;
        let chunk1_end = writer.inner().get_ref().len();
        writer.write_message(EncoderMessage::Array(&chunk2)).await?;
        let data = writer.into_inner().into_inner();

        let poll_calls = Arc::new(AtomicUsize::new(0));
        let mut reader = AsyncIPCReader::try_new(
            FailAfterAsyncReader {
                data,
                pos: 0,
                fail_at: chunk1_end + 2,
                poll_calls: Arc::clone(&poll_calls),
            },
            &SESSION,
        )
        .await?;

        let first = reader.next().await.vortex_expect("expected first chunk")?;
        assert_arrays_eq!(first, chunk1, &mut SESSION.create_execution_ctx());

        let Err(err) = reader.next().await.vortex_expect("expected io error") else {
            vortex_panic!("expected io error");
        };
        assert_io_error_kind(&err, io::ErrorKind::BrokenPipe);

        let calls_after_err = poll_calls.load(Ordering::SeqCst);
        assert!(reader.next().await.is_none());
        assert_eq!(poll_calls.load(Ordering::SeqCst), calls_after_err);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_ipc_reader_rejects_non_array_message() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        let raw_buf = ByteBuffer::copy_from([1u8, 2, 3, 4]);
        let dtype_msg = EncoderMessage::DType(array.dtype());
        let buf_msg = EncoderMessage::Buffer(&raw_buf);
        let mut writer = AsyncMessageWriter::new(Cursor::new(Vec::new()), &SESSION);
        writer.write_message(dtype_msg).await?;
        writer.write_message(buf_msg).await?;
        writer.write_message(EncoderMessage::Array(&array)).await?;
        let data = writer.into_inner().into_inner();

        let mut reader = AsyncIPCReader::try_new(Cursor::new(data), &SESSION).await?;
        let Err(err) = reader.next().await.vortex_expect("non-array message") else {
            vortex_panic!("expected error on non-array message");
        };
        assert!(err.to_string().contains("Expected Array message"));
        assert!(reader.next().await.is_none());
        Ok(())
    }
}

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

        match ready!(this.reader.poll_next(cx)) {
            None => Poll::Ready(None),
            Some(msg) => match msg {
                Ok(DecoderMessage::Array((array_parts, ctx, row_count))) => Poll::Ready(Some(
                    array_parts
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

        // If we haven't written the dtype yet, we write it
        if !this.written_dtype {
            let Ok(buffers) = this
                .encoder
                .encode(EncoderMessage::DType(this.stream.dtype()))
            else {
                return Poll::Ready(Some(Err(vortex_err!("Failed to encode DType message"))));
            };
            this.buffers.extend(buffers);
            this.written_dtype = true;
        }

        // Try to flush any buffers we have
        if !this.buffers.is_empty() {
            return Poll::Ready(Some(Ok(this.buffers.remove(0))));
        }

        // Or else try to serialize the next array
        match ready!(this.stream.poll_next_unpin(cx)) {
            None => return Poll::Ready(None),
            Some(chunk) => match chunk.and_then(|c| this.encoder.encode(EncoderMessage::Array(&c)))
            {
                Ok(buffers) => {
                    this.buffers.extend(buffers);
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
    use std::error::Error as _;
    use std::io;
    use std::pin::Pin;
    use std::task::Context;
    use std::task::Poll;

    use futures::io::Cursor;
    use vortex_array::IntoArray as _;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::iter::ArrayIterator;
    use vortex_array::stream::ArrayStream;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_buffer::buffer;
    use vortex_error::VortexError;
    use vortex_error::VortexExpect;

    use super::*;
    use crate::iterator::SyncIPCReader;
    use crate::test::SESSION;

    #[tokio::test]
    async fn test_async_stream() {
        let array = buffer![1, 2, 3].into_array();
        let ipc_buffer = array
            .to_array_stream()
            .into_ipc(&SESSION)
            .collect_to_buffer()
            .await
            .unwrap();

        let reader = AsyncIPCReader::try_new(Cursor::new(ipc_buffer), &SESSION)
            .await
            .unwrap();

        assert_eq!(reader.dtype(), array.dtype());
        let result = reader.read_all().await.unwrap();
        assert_arrays_eq!(result, array, &mut SESSION.create_execution_ctx());
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
    async fn test_async_stream_chunked() {
        let array = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array();
        let ipc_buffer = array
            .to_array_stream()
            .into_ipc(&SESSION)
            .collect_to_buffer()
            .await
            .unwrap();

        let chunked = ChunkedReader {
            inner: Cursor::new(ipc_buffer),
            chunk_size: 3,
        };

        let reader = AsyncIPCReader::try_new(chunked, &SESSION).await.unwrap();

        let result = reader.read_all().await.unwrap();
        let expected = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
    }

    /// Test with 1-byte chunks to stress-test partial read handling.
    #[tokio::test]
    async fn test_async_stream_single_byte_chunks() {
        let array = buffer![42i64, -1, 0, i64::MAX, i64::MIN].into_array();
        let ipc_buffer = array
            .to_array_stream()
            .into_ipc(&SESSION)
            .collect_to_buffer()
            .await
            .unwrap();

        let chunked = ChunkedReader {
            inner: Cursor::new(ipc_buffer),
            chunk_size: 1,
        };

        let reader = AsyncIPCReader::try_new(chunked, &SESSION).await.unwrap();

        let result = reader.read_all().await.unwrap();
        let expected = buffer![42i64, -1, 0, i64::MAX, i64::MIN].into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
    }

    fn assert_unexpected_eof(err: &VortexError) {
        let io_err = err
            .source()
            .and_then(|s| s.downcast_ref::<io::Error>())
            .vortex_expect("expected io::Error source");
        assert_eq!(io_err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[tokio::test]
    async fn sync_and_async_adapters_match_on_complete_and_truncated_streams() -> VortexResult<()> {
        let chunk1 = buffer![1i32, 2, 3].into_array();
        let chunk2 = buffer![4i32, 5, 6, 7].into_array();

        let mut encoder = MessageEncoder::new(SESSION.clone());
        let dtype_parts = encoder.encode(EncoderMessage::DType(chunk1.dtype()))?;
        let chunk1_parts = encoder.encode(EncoderMessage::Array(&chunk1))?;
        let chunk2_parts = encoder.encode(EncoderMessage::Array(&chunk2))?;

        let mut full = Vec::new();
        for part in &dtype_parts {
            full.extend_from_slice(part.as_ref());
        }
        let after_dtype = full.len();
        for part in &chunk1_parts {
            full.extend_from_slice(part.as_ref());
        }
        let after_chunk1 = full.len();
        let chunk2_header_len = chunk2_parts[1].len();
        let chunk2_body_len: usize = chunk2_parts[2..].iter().map(|b| b.len()).sum();
        for part in &chunk2_parts {
            full.extend_from_slice(part.as_ref());
        }

        // Complete multi-message stream: both adapters return identical values.
        let sync_reader = SyncIPCReader::try_new(io::Cursor::new(full.clone()), &SESSION)?;
        let async_reader = AsyncIPCReader::try_new(Cursor::new(full.clone()), &SESSION).await?;
        assert_eq!(sync_reader.dtype(), async_reader.dtype());

        let sync_chunks: Vec<ArrayRef> = sync_reader.collect::<VortexResult<_>>()?;
        let async_chunks: Vec<ArrayRef> = async_reader.try_collect().await?;
        assert_eq!(sync_chunks.len(), 2);
        assert_eq!(async_chunks.len(), 2);
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(sync_chunks[0], async_chunks[0], &mut ctx);
        assert_arrays_eq!(sync_chunks[1], async_chunks[1], &mut ctx);
        assert_arrays_eq!(sync_chunks[0], chunk1, &mut ctx);
        assert_arrays_eq!(sync_chunks[1], chunk2, &mut ctx);

        // Initial dtype contract: empty stream yields the dtype EOF error, not UnexpectedEof.
        let sync_empty_err = match SyncIPCReader::try_new(io::Cursor::new(Vec::new()), &SESSION) {
            Ok(_) => unreachable!("empty stream must fail try_new"),
            Err(e) => e,
        };
        assert!(
            sync_empty_err
                .to_string()
                .contains("Expected DType message, got EOF")
        );
        let async_empty_err = match AsyncIPCReader::try_new(Cursor::new(Vec::new()), &SESSION).await
        {
            Ok(_) => unreachable!("empty stream must fail try_new"),
            Err(e) => e,
        };
        assert!(
            async_empty_err
                .to_string()
                .contains("Expected DType message, got EOF")
        );

        // Truncated DType message fails try_new with UnexpectedEof.
        let dtype_cut = 4;
        let sync_dtype_err =
            match SyncIPCReader::try_new(io::Cursor::new(full[..dtype_cut].to_vec()), &SESSION) {
                Ok(_) => unreachable!("truncated dtype must fail try_new"),
                Err(e) => e,
            };
        assert_unexpected_eof(&sync_dtype_err);
        let async_dtype_err = match AsyncIPCReader::try_new(
            Cursor::new(full[..dtype_cut].to_vec()),
            &SESSION,
        )
        .await
        {
            Ok(_) => unreachable!("truncated dtype must fail try_new"),
            Err(e) => e,
        };
        assert_unexpected_eof(&async_dtype_err);

        // Truncated after dtype (before chunk1 finishes) and after chunk1 (before chunk2 finishes):
        // exact phase boundaries (after length, after header) and mid-body.
        let chunk2_cuts = [
            after_chunk1 + 2,
            after_chunk1 + 4,
            after_chunk1 + 4 + chunk2_header_len,
            after_chunk1 + 4 + chunk2_header_len + chunk2_body_len / 2,
        ];
        for cut in chunk2_cuts {
            let truncated = full[..cut].to_vec();

            let mut sync_reader =
                SyncIPCReader::try_new(io::Cursor::new(truncated.clone()), &SESSION)?;
            let first = sync_reader.next().vortex_expect("expected first chunk")?;
            assert_arrays_eq!(first, chunk1, &mut ctx);
            let err = sync_reader
                .next()
                .vortex_expect("expected error on truncated second chunk")
                .unwrap_err();
            assert_unexpected_eof(&err);
            assert!(sync_reader.next().is_none());

            let mut async_reader =
                AsyncIPCReader::try_new(Cursor::new(truncated), &SESSION).await?;
            let first = async_reader
                .next()
                .await
                .vortex_expect("expected first chunk")?;
            assert_arrays_eq!(first, chunk1, &mut ctx);
            let err = async_reader
                .next()
                .await
                .vortex_expect("expected error on truncated second chunk")
                .unwrap_err();
            assert_unexpected_eof(&err);
            assert!(async_reader.next().await.is_none());
        }

        // Clean empty array stream right after dtype still ends cleanly.
        let mut sync_after_dtype =
            SyncIPCReader::try_new(io::Cursor::new(full[..after_dtype].to_vec()), &SESSION)?;
        assert!(sync_after_dtype.next().is_none());
        let mut async_after_dtype =
            AsyncIPCReader::try_new(Cursor::new(full[..after_dtype].to_vec()), &SESSION).await?;
        assert!(async_after_dtype.next().await.is_none());

        Ok(())
    }
}

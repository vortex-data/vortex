// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Read;
use std::io::Write;

use bytes::Bytes;
use bytes::BytesMut;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::iter::ArrayIterator;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::messages::DecoderMessage;
use crate::messages::EncoderMessage;
use crate::messages::MessageEncoder;
use crate::messages::SyncMessageReader;

/// An [`ArrayIterator`] for reading messages off an IPC stream.
pub struct SyncIPCReader<R: Read> {
    reader: SyncMessageReader<R>,
    dtype: DType,
    session: VortexSession,
    done: bool,
}

impl<R: Read> SyncIPCReader<R> {
    pub fn try_new(read: R, session: &VortexSession) -> VortexResult<Self> {
        let mut reader = SyncMessageReader::new(read);
        match reader.next().transpose()? {
            Some(msg) => match msg {
                DecoderMessage::DType(fb_dtype) => {
                    let dtype = DType::from_flatbuffer(fb_dtype, session)?;
                    Ok(SyncIPCReader {
                        reader,
                        dtype,
                        session: session.clone(),
                        done: false,
                    })
                }
                msg => {
                    vortex_bail!("Expected DType message, got {:?}", msg);
                }
            },
            None => vortex_bail!("Expected DType message, got EOF"),
        }
    }
}

impl<R: Read> ArrayIterator for SyncIPCReader<R> {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl<R: Read> Iterator for SyncIPCReader<R> {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let Some(msg) = self.reader.next() else {
            self.done = true;
            return None;
        };
        let result = match msg {
            Ok(DecoderMessage::Array((array_parts, ctx, row_count))) => array_parts
                .decode(&self.dtype, row_count, &ctx, &self.session)
                .and_then(|array| {
                    if array.dtype() != self.dtype() {
                        Err(vortex_err!(
                            "Array data type mismatch: expected {:?}, got {:?}",
                            self.dtype(),
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
            self.done = true;
        }
        Some(result)
    }
}

/// A trait for converting an [`ArrayIterator`] into an IPC stream.
pub trait ArrayIteratorIPC {
    fn into_ipc(self, session: &VortexSession) -> VortexResult<ArrayIteratorIPCBytes>
    where
        Self: Sized;

    fn write_ipc<W: Write>(self, write: W, session: &VortexSession) -> VortexResult<W>
    where
        Self: Sized;
}

impl<I: ArrayIterator + 'static> ArrayIteratorIPC for I {
    fn into_ipc(self, session: &VortexSession) -> VortexResult<ArrayIteratorIPCBytes>
    where
        Self: Sized,
    {
        let mut encoder = MessageEncoder::new(session.clone());
        let buffers = encoder.encode(EncoderMessage::DType(self.dtype()))?;
        Ok(ArrayIteratorIPCBytes {
            inner: Box::new(self),
            encoder,
            buffers,
            done: false,
        })
    }

    fn write_ipc<W: Write>(self, mut write: W, session: &VortexSession) -> VortexResult<W>
    where
        Self: Sized,
    {
        let mut stream = self.into_ipc(session)?;
        for buffer in &mut stream {
            write.write_all(buffer?.as_ref())?;
        }
        Ok(write)
    }
}

pub struct ArrayIteratorIPCBytes {
    inner: Box<dyn ArrayIterator + 'static>,
    encoder: MessageEncoder,
    buffers: Vec<Bytes>,
    done: bool,
}

impl ArrayIteratorIPCBytes {
    /// Collects the IPC bytes into a single `Bytes`.
    pub fn collect_to_buffer(self) -> VortexResult<Bytes> {
        let buffers: Vec<Bytes> = self.try_collect()?;
        let mut buffer = BytesMut::with_capacity(buffers.iter().map(|b| b.len()).sum());
        for buf in buffers {
            buffer.extend_from_slice(buf.as_ref());
        }
        Ok(buffer.freeze())
    }
}

impl Iterator for ArrayIteratorIPCBytes {
    type Item = VortexResult<Bytes>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Try to flush any buffers we have
        if !self.buffers.is_empty() {
            return Some(Ok(self.buffers.remove(0)));
        }

        // Or else try to serialize the next array
        let Some(chunk) = self.inner.next() else {
            self.done = true;
            return None;
        };
        match chunk.and_then(|c| self.encoder.encode(EncoderMessage::Array(&c))) {
            Ok(buffers) => self.buffers.extend(buffers),
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        }

        // Try to flush any buffers we have again
        if !self.buffers.is_empty() {
            return Some(Ok(self.buffers.remove(0)));
        }

        // Otherwise, we're done
        self.done = true;
        None
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::io::Cursor;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::vec::IntoIter;

    use vortex_array::IntoArray as _;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::iter::ArrayIteratorAdapter;
    use vortex_array::iter::ArrayIteratorExt;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::vortex_panic;

    use super::*;
    use crate::messages::SyncMessageWriter;
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

    #[test]
    fn test_sync_stream() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        let ipc_buffer = array
            .to_array_iterator()
            .into_ipc(&SESSION)?
            .collect_to_buffer()?;

        let reader = SyncIPCReader::try_new(Cursor::new(ipc_buffer), &SESSION)?;

        assert_eq!(reader.dtype(), array.dtype());
        let result = reader.read_all()?;
        assert_arrays_eq!(result, array, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn test_sync_write_ipc_roundtrip() -> VortexResult<()> {
        let chunk1 = buffer![1i32, 2, 3].into_array();
        let chunk2 = buffer![4i32, 5, 6, 7].into_array();
        let dtype = chunk1.dtype().clone();
        let expected = buffer![1i32, 2, 3, 4, 5, 6, 7].into_array();

        let bytes = ArrayIteratorAdapter::new(dtype, [Ok(chunk1), Ok(chunk2)].into_iter())
            .write_ipc(Vec::new(), &SESSION)?;
        let reader = SyncIPCReader::try_new(Cursor::new(bytes), &SESSION)?;

        assert_eq!(reader.dtype(), expected.dtype());
        let result = reader.read_all()?;
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[test]
    fn test_sync_ipc_bytes_mid_stream_error_terminates() -> VortexResult<()> {
        let chunk1 = buffer![1i32, 2].into_array();
        let chunk2 = buffer![3i32, 4].into_array();
        let dtype = chunk1.dtype().clone();
        let pulled = Arc::new(AtomicUsize::new(0));
        let items = vec![
            Ok(chunk1.clone()),
            Err(vortex_err!("boom")),
            Ok(chunk2.clone()),
        ];

        let iter = ArrayIteratorAdapter::new(
            dtype.clone(),
            CountingIter {
                items: items.into_iter(),
                pulled: Arc::clone(&pulled),
            },
        );
        let mut ipc_bytes = iter.into_ipc(&SESSION)?;

        let mut seen_bytes = 0usize;
        let err = loop {
            match ipc_bytes.next().vortex_expect("expected mid-stream error") {
                Ok(buf) => seen_bytes += buf.len(),
                Err(err) => break err,
            }
        };
        assert!(seen_bytes > 0);
        assert!(err.to_string().contains("boom"));
        assert_eq!(pulled.load(Ordering::SeqCst), 2);
        assert!(ipc_bytes.next().is_none());
        assert_eq!(pulled.load(Ordering::SeqCst), 2);

        let err_iter = ArrayIteratorAdapter::new(
            dtype,
            [Ok(chunk1), Err(vortex_err!("boom")), Ok(chunk2)].into_iter(),
        );
        assert!(err_iter.into_ipc(&SESSION)?.collect_to_buffer().is_err());
        Ok(())
    }

    #[test]
    fn test_sync_ipc_reader_mid_stream_io_error_terminates() -> VortexResult<()> {
        struct FailAfterReader {
            data: Vec<u8>,
            pos: usize,
            fail_at: usize,
            read_calls: Arc<AtomicUsize>,
        }

        impl Read for FailAfterReader {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                self.read_calls.fetch_add(1, Ordering::SeqCst);
                if self.pos >= self.fail_at {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "mid-stream failure",
                    ));
                }
                let remaining = self.fail_at - self.pos;
                let count = buf.len().min(remaining);
                buf[..count].copy_from_slice(&self.data[self.pos..self.pos + count]);
                self.pos += count;
                Ok(count)
            }
        }

        let chunk1 = buffer![1i32, 2, 3].into_array();
        let chunk2 = buffer![4i32, 5, 6].into_array();
        let mut data = Vec::new();
        {
            let mut writer = SyncMessageWriter::new(&mut data, &SESSION);
            writer.write_message(EncoderMessage::DType(chunk1.dtype()))?;
            writer.write_message(EncoderMessage::Array(&chunk1))?;
        }
        let chunk1_end = data.len();
        SyncMessageWriter::new(&mut data, &SESSION)
            .write_message(EncoderMessage::Array(&chunk2))?;

        let read_calls = Arc::new(AtomicUsize::new(0));
        let mut reader = SyncIPCReader::try_new(
            FailAfterReader {
                data,
                pos: 0,
                fail_at: chunk1_end + 2,
                read_calls: Arc::clone(&read_calls),
            },
            &SESSION,
        )?;

        let first = reader.next().vortex_expect("expected first chunk")?;
        assert_arrays_eq!(first, chunk1, &mut SESSION.create_execution_ctx());

        let Err(err) = reader.next().vortex_expect("expected io error") else {
            vortex_panic!("expected io error");
        };
        assert_io_error_kind(&err, io::ErrorKind::BrokenPipe);

        let calls_after_err = read_calls.load(Ordering::SeqCst);
        assert!(reader.next().is_none());
        assert_eq!(read_calls.load(Ordering::SeqCst), calls_after_err);
        Ok(())
    }

    #[test]
    fn test_sync_ipc_reader_rejects_non_array_message() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        let raw_buf = ByteBuffer::copy_from([1u8, 2, 3, 4]);
        let mut data = Vec::new();
        let mut writer = SyncMessageWriter::new(&mut data, &SESSION);
        writer.write_message(EncoderMessage::DType(array.dtype()))?;
        writer.write_message(EncoderMessage::Buffer(&raw_buf))?;
        writer.write_message(EncoderMessage::Array(&array))?;

        let mut reader = SyncIPCReader::try_new(Cursor::new(data), &SESSION)?;
        let Err(err) = reader.next().vortex_expect("non-array message") else {
            vortex_panic!("expected error on non-array message");
        };
        assert!(err.to_string().contains("Expected Array message"));
        assert!(reader.next().is_none());
        Ok(())
    }
}

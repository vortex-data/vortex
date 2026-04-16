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
        match self.reader.next()? {
            Ok(msg) => match msg {
                DecoderMessage::Array((array_parts, ctx, row_count)) => Some(
                    array_parts
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
                ),
                msg => Some(Err(vortex_err!("Expected Array message, got {:?}", msg))),
            },
            Err(e) => Some(Err(e)),
        }
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
        // Try to flush any buffers we have
        if !self.buffers.is_empty() {
            return Some(Ok(self.buffers.remove(0)));
        }

        // Or else try to serialize the next array
        match self
            .inner
            .next()?
            .and_then(|chunk| self.encoder.encode(EncoderMessage::Array(&chunk)))
        {
            Ok(buffers) => self.buffers.extend(buffers),
            Err(e) => return Some(Err(e)),
        }

        // Try to flush any buffers we have again
        if !self.buffers.is_empty() {
            return Some(Ok(self.buffers.remove(0)));
        }

        // Otherwise, we're done
        None
    }
}

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use vortex_array::IntoArray as _;
    use vortex_array::assert_arrays_eq;
    use vortex_array::iter::ArrayIterator;
    use vortex_array::iter::ArrayIteratorExt;
    use vortex_buffer::buffer;

    use super::*;
    use crate::test::SESSION;

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
        assert_arrays_eq!(result, array);

        Ok(())
    }
}

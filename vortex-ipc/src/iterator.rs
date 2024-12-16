use std::io::{Read, Write};
use std::sync::Arc;

use vortex_array::iter::ArrayIterator;
use vortex_array::{ArrayDType, ArrayData, Context};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::messages::{DecoderMessage, EncoderMessage, MessageEncoder, SyncMessageReader};
use crate::ALIGNMENT;

/// An [`ArrayIterator`] for reading messages off an IPC stream.
pub struct SyncIPCReader<R: Read> {
    reader: SyncMessageReader<R>,
    ctx: Arc<Context>,
    dtype: DType,
}

impl<R: Read> SyncIPCReader<R> {
    pub fn try_new(read: R, ctx: Arc<Context>) -> VortexResult<Self> {
        let mut reader = SyncMessageReader::new(read);
        match reader.read_message()? {
            Some(msg) => match msg {
                DecoderMessage::DType(dtype) => Ok(SyncIPCReader { reader, ctx, dtype }),
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
    type Item = VortexResult<ArrayData>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.read_message().transpose()? {
            Ok(msg) => match msg {
                DecoderMessage::Array(array_parts) => Some(
                    array_parts
                        .into_array_data(self.ctx.clone(), self.dtype.clone())
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
    fn into_ipc(self) -> ArrayIteratorIPCBytes
    where
        Self: Sized;

    fn write_ipc<W: Write>(self, write: W) -> VortexResult<W>
    where
        Self: Sized;
}

impl<I: ArrayIterator + 'static> ArrayIteratorIPC for I {
    fn into_ipc(self) -> ArrayIteratorIPCBytes
    where
        Self: Sized,
    {
        let mut encoder = MessageEncoder::new(ALIGNMENT);
        let buffers = encoder.encode(EncoderMessage::DType(self.dtype()));
        ArrayIteratorIPCBytes {
            inner: Box::new(self),
            encoder,
            buffers,
        }
    }

    fn write_ipc<W: Write>(self, mut write: W) -> VortexResult<W>
    where
        Self: Sized,
    {
        let mut stream = self.into_ipc();
        for buffer in &mut stream {
            write.write_all(buffer?.as_slice())?;
        }
        Ok(write)
    }
}

pub struct ArrayIteratorIPCBytes {
    inner: Box<dyn ArrayIterator + 'static>,
    encoder: MessageEncoder,
    buffers: Vec<Buffer>,
}

impl ArrayIteratorIPCBytes {
    /// Collects the IPC bytes into a single `Vec<u8>` buffer.
    pub fn collect_to_buffer(self) -> VortexResult<Vec<u8>> {
        // TODO(ngates): this buffer needs 64-byte alignment...
        let mut buffer = vec![];
        for chunk in self {
            buffer.extend(chunk?.as_slice());
        }
        Ok(buffer)
    }
}

impl Iterator for ArrayIteratorIPCBytes {
    type Item = VortexResult<Buffer>;

    fn next(&mut self) -> Option<Self::Item> {
        // Try to flush any buffers we have
        if !self.buffers.is_empty() {
            return Some(Ok(self.buffers.remove(0)));
        }

        // Or else try to serialize the next array
        match self.inner.next()? {
            Ok(chunk) => {
                self.buffers
                    .extend(self.encoder.encode(EncoderMessage::Array(&chunk)));
            }
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
    use std::sync::Arc;

    use vortex_array::array::PrimitiveArray;
    use vortex_array::iter::{ArrayIterator, ArrayIteratorExt};
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayDType, Context, IntoArrayVariant, ToArrayData};

    use super::*;

    #[test]
    fn test_sync_stream() {
        let array = PrimitiveArray::from_vec::<i32>(vec![1, 2, 3], Validity::NonNullable);
        let ipc_buffer = array
            .to_array()
            .into_array_iterator()
            .into_ipc()
            .collect_to_buffer()
            .unwrap();

        let reader =
            SyncIPCReader::try_new(Cursor::new(ipc_buffer), Arc::new(Context::default())).unwrap();

        assert_eq!(reader.dtype(), array.dtype());
        let result = reader.into_array_data().unwrap().into_primitive().unwrap();
        assert_eq!(
            array.maybe_null_slice::<i32>(),
            result.maybe_null_slice::<i32>()
        );
    }
}

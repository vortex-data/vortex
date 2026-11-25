// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::AsyncWrite;
use futures::AsyncWriteExt;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::messages::EncoderMessage;
use crate::messages::MessageEncoder;

pub struct AsyncMessageWriter<W> {
    write: W,
    encoder: MessageEncoder,
}

impl<W: AsyncWrite + Unpin> AsyncMessageWriter<W> {
    pub fn new(write: W, session: &VortexSession) -> Self {
        Self {
            write,
            encoder: MessageEncoder::new(session.clone()),
        }
    }

    pub async fn write_message(&mut self, message: EncoderMessage<'_>) -> VortexResult<()> {
        for buffer in self.encoder.encode(message)? {
            self.write.write_all(&buffer).await?;
        }
        Ok(())
    }

    pub fn inner(&self) -> &W {
        &self.write
    }

    pub fn into_inner(self) -> W {
        self.write
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use futures::io::Cursor;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::vortex_panic;

    use super::*;
    use crate::messages::AsyncMessageReader;
    use crate::messages::DecoderMessage;
    use crate::tests::SESSION;

    #[tokio::test]
    async fn async_writer_reader_roundtrip() -> VortexResult<()> {
        let array = buffer![10i32, 20, 30, 40].into_array();
        let raw_buf = ByteBuffer::copy_from_aligned([2u8, 7, 1, 8, 2, 8], Alignment::new(64));

        let dtype_msg = EncoderMessage::DType(array.dtype());
        let buf_msg = EncoderMessage::Buffer(&raw_buf);
        let mut writer = AsyncMessageWriter::new(Cursor::new(Vec::new()), &SESSION);
        writer.write_message(dtype_msg).await?;
        writer.write_message(buf_msg).await?;
        writer.write_message(EncoderMessage::Array(&array)).await?;
        assert!(!writer.inner().get_ref().is_empty());

        let bytes = writer.into_inner().into_inner();
        let mut reader = AsyncMessageReader::new(Cursor::new(bytes));

        let DecoderMessage::DType(fb_dtype) = reader.next().await.vortex_expect("dtype message")?
        else {
            vortex_panic!("expected a dtype message");
        };
        assert_eq!(DType::from_flatbuffer(fb_dtype, &SESSION)?, *array.dtype());

        let DecoderMessage::Buffer(decoded_buf) =
            reader.next().await.vortex_expect("buffer message")?
        else {
            vortex_panic!("expected a buffer message");
        };
        assert_eq!(decoded_buf.as_slice(), raw_buf.as_slice());
        assert_eq!(decoded_buf.alignment(), Alignment::new(64));
        assert!(Alignment::new(64).is_ptr_aligned(decoded_buf.as_ptr()));

        let DecoderMessage::Array((parts, ctx, row_count)) =
            reader.next().await.vortex_expect("array message")?
        else {
            vortex_panic!("expected an array message");
        };
        let actual = parts.decode(array.dtype(), row_count, &ctx, &SESSION)?;
        assert_arrays_eq!(actual, array, &mut SESSION.create_execution_ctx());

        assert!(reader.next().await.is_none());
        Ok(())
    }
}

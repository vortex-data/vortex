// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::messages::EncoderMessage;
use crate::messages::MessageEncoder;

pub struct SyncMessageWriter<W> {
    write: W,
    encoder: MessageEncoder,
}

impl<W: Write> SyncMessageWriter<W> {
    pub fn new(write: W, session: &VortexSession) -> Self {
        Self {
            write,
            encoder: MessageEncoder::new(session.clone()),
        }
    }

    pub fn write_message(&mut self, message: EncoderMessage) -> VortexResult<()> {
        for buffer in self.encoder.encode(message)? {
            self.write.write_all(&buffer)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

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
    use crate::messages::DecoderMessage;
    use crate::messages::SyncMessageReader;
    use crate::tests::SESSION;

    #[test]
    fn sync_writer_reader_roundtrip() -> VortexResult<()> {
        let array = buffer![10i32, 20, 30, 40].into_array();
        let raw_buf = ByteBuffer::copy_from_aligned([3u8, 1, 4, 1, 5, 9], Alignment::new(64));

        let mut bytes = Vec::new();
        let mut writer = SyncMessageWriter::new(&mut bytes, &SESSION);
        writer.write_message(EncoderMessage::DType(array.dtype()))?;
        writer.write_message(EncoderMessage::Buffer(&raw_buf))?;
        writer.write_message(EncoderMessage::Array(&array))?;

        let mut reader = SyncMessageReader::new(Cursor::new(bytes));

        let DecoderMessage::DType(fb_dtype) =
            reader.next().vortex_expect("expected dtype message")?
        else {
            vortex_panic!("expected a dtype message");
        };
        assert_eq!(DType::from_flatbuffer(fb_dtype, &SESSION)?, *array.dtype());

        let DecoderMessage::Buffer(decoded_buf) =
            reader.next().vortex_expect("expected buffer message")?
        else {
            vortex_panic!("expected a buffer message");
        };
        assert_eq!(decoded_buf.as_slice(), raw_buf.as_slice());
        assert_eq!(decoded_buf.alignment(), Alignment::new(64));
        assert!(Alignment::new(64).is_ptr_aligned(decoded_buf.as_ptr()));

        let DecoderMessage::Array((parts, ctx, row_count)) =
            reader.next().vortex_expect("expected array message")?
        else {
            vortex_panic!("expected an array message");
        };
        let actual = parts.decode(array.dtype(), row_count, &ctx, &SESSION)?;
        assert_arrays_eq!(actual, array, &mut SESSION.create_execution_ctx());

        assert!(reader.next().is_none());
        Ok(())
    }
}

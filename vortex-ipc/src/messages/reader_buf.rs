// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bytes::Buf;
use vortex_error::VortexResult;

use crate::messages::DecoderMessage;
use crate::messages::MessageDecoder;
use crate::messages::PollRead;

/// An IPC message reader backed by a `Buf`.
pub struct BufMessageReader<B> {
    buffer: B,
    decoder: MessageDecoder,
    done: bool,
}

impl<B: Buf> BufMessageReader<B> {
    pub fn new(buffer: B) -> Self {
        BufMessageReader {
            buffer,
            decoder: MessageDecoder::default(),
            done: false,
        }
    }
}

impl<B: Buf> Iterator for BufMessageReader<B> {
    type Item = VortexResult<DecoderMessage>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.decoder.read_next(&mut self.buffer) {
            Ok(PollRead::Some(msg)) => Some(Ok(msg)),
            Ok(PollRead::NeedMore(_)) => {
                self.done = true;
                match self.decoder.finish_eof(self.buffer.remaining()) {
                    Ok(()) => None,
                    Err(e) => Some(Err(e)),
                }
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use bytes::Buf;
    use bytes::Bytes;
    use bytes::BytesMut;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_panic;

    use super::*;
    use crate::messages::EncoderMessage;
    use crate::messages::MessageEncoder;
    use crate::tests::SESSION;
    use crate::tests::assert_io_error_kind;

    struct ChunkedBuf {
        data: Vec<u8>,
        pos: usize,
        chunk_size: usize,
    }

    impl ChunkedBuf {
        fn new(data: Vec<u8>, chunk_size: usize) -> Self {
            Self {
                data,
                pos: 0,
                chunk_size,
            }
        }
    }

    impl Buf for ChunkedBuf {
        fn remaining(&self) -> usize {
            self.data.len() - self.pos
        }

        fn chunk(&self) -> &[u8] {
            let end = (self.pos + self.chunk_size).min(self.data.len());
            &self.data[self.pos..end]
        }

        fn advance(&mut self, cnt: usize) {
            assert!(cnt <= self.remaining());
            self.pos += cnt;
        }
    }

    #[test]
    fn empty_buffer_terminates_cleanly() {
        let mut reader = BufMessageReader::new(Bytes::new());
        assert!(reader.next().is_none());
        assert!(reader.next().is_none());
    }

    #[test]
    fn reads_messages_across_small_chunk_sizes() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3, 4, 5].into_array();
        let raw_buf = ByteBuffer::copy_from_aligned([1u8, 2, 3, 4, 5, 6, 7], Alignment::new(64));
        let mut ipc_bytes = BytesMut::new();
        let mut encoder = MessageEncoder::new(SESSION.clone());
        for message in [
            EncoderMessage::DType(array.dtype()),
            EncoderMessage::Array(&array),
            EncoderMessage::Buffer(&raw_buf),
        ] {
            for buf in encoder.encode(message)? {
                ipc_bytes.extend_from_slice(buf.as_ref());
            }
        }
        let encoded = ipc_bytes.to_vec();

        for chunk_size in [1, 2, 3, 7, 13] {
            let mut reader = BufMessageReader::new(ChunkedBuf::new(encoded.clone(), chunk_size));

            let DecoderMessage::DType(fb_dtype) =
                reader.next().vortex_expect("expected dtype message")?
            else {
                vortex_panic!("expected a dtype message");
            };
            assert_eq!(DType::from_flatbuffer(fb_dtype, &SESSION)?, *array.dtype());

            let DecoderMessage::Array((parts, ctx, row_count)) =
                reader.next().vortex_expect("expected array message")?
            else {
                vortex_panic!("expected an array message");
            };
            let actual = parts.decode(array.dtype(), row_count, &ctx, &SESSION)?;
            assert_arrays_eq!(actual, array, &mut SESSION.create_execution_ctx());

            let DecoderMessage::Buffer(decoded_buf) =
                reader.next().vortex_expect("expected buffer message")?
            else {
                vortex_panic!("expected a buffer message");
            };
            assert_eq!(decoded_buf.as_slice(), raw_buf.as_slice());
            assert_eq!(decoded_buf.alignment(), Alignment::new(64));
            assert!(Alignment::new(64).is_ptr_aligned(decoded_buf.as_ptr()));

            assert!(reader.next().is_none());
        }

        Ok(())
    }

    #[test]
    fn truncated_buffer_errors_once_then_terminates() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        let mut encoder = MessageEncoder::new(SESSION.clone());
        let parts = encoder.encode(EncoderMessage::Array(&array))?;
        let header_len = parts[1].len();
        let body_len: usize = parts[2..].iter().map(|b| b.len()).sum();
        let full: Vec<u8> = parts.iter().flat_map(|b| b.as_ref()).copied().collect();

        for cut in [
            2,
            4,
            4 + header_len / 2,
            4 + header_len,
            4 + header_len + body_len / 2,
        ] {
            let mut reader = BufMessageReader::new(Bytes::copy_from_slice(&full[..cut]));
            let Err(err) = reader.next().vortex_expect("truncated buffer") else {
                vortex_panic!("expected error on truncated buffer");
            };
            assert_io_error_kind(&err, io::ErrorKind::UnexpectedEof);
            assert!(reader.next().is_none());
            assert!(reader.next().is_none());
        }

        Ok(())
    }
}

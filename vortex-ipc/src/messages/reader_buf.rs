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

    use bytes::Bytes;
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use super::*;
    use crate::messages::EncoderMessage;
    use crate::messages::MessageEncoder;
    use crate::test::SESSION;
    use crate::test::assert_io_error_kind;

    #[test]
    fn truncated_buffer_errors_once_then_terminates() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        let mut encoder = MessageEncoder::new(SESSION.clone());
        let parts = encoder.encode(EncoderMessage::Array(&array))?;
        let header_len = parts[1].len();
        let body_len: usize = parts[2..].iter().map(|b| b.len()).sum();
        let full: Vec<u8> = parts.iter().flat_map(|b| b.as_ref()).copied().collect();

        for cut in [2, 4, 4 + header_len, 4 + header_len + body_len / 2] {
            let mut reader = BufMessageReader::new(Bytes::copy_from_slice(&full[..cut]));
            let err = reader
                .next()
                .vortex_expect("expected error on truncated buffer")
                .unwrap_err();
            assert_io_error_kind(&err, io::ErrorKind::UnexpectedEof);
            assert!(reader.next().is_none());
            assert!(reader.next().is_none());
        }

        Ok(())
    }
}

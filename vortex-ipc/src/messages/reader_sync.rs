// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Read;

use bytes::BytesMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::messages::DecoderMessage;
use crate::messages::MessageDecoder;
use crate::messages::PollRead;

/// An IPC message reader backed by a `Read` stream.
pub struct SyncMessageReader<R> {
    read: R,
    buffer: BytesMut,
    decoder: MessageDecoder,
}

impl<R: Read> SyncMessageReader<R> {
    pub fn new(read: R) -> Self {
        SyncMessageReader {
            read,
            buffer: BytesMut::new(),
            decoder: MessageDecoder::default(),
        }
    }
}

impl<R: Read> Iterator for SyncMessageReader<R> {
    type Item = VortexResult<DecoderMessage>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.decoder.read_next(&mut self.buffer) {
                Ok(PollRead::Some(msg)) => {
                    return Some(Ok(msg));
                }
                Ok(PollRead::NeedMore(nbytes)) => {
                    self.buffer.resize(nbytes, 0x00);
                    // `Read` can return fewer bytes than requested; fill completely
                    // before decoding, which reads `nbytes` regardless of how many
                    // were actually written.
                    let mut total_bytes_read = 0;
                    while total_bytes_read < nbytes {
                        match self.read.read(&mut self.buffer[total_bytes_read..]) {
                            Ok(0) => {
                                if total_bytes_read == 0 {
                                    // Clean EOF at a message boundary.
                                    return None;
                                }
                                return Some(Err(vortex_err!(
                                    "unexpected EOF during partial read: read {total_bytes_read} of {nbytes} expected bytes"
                                )));
                            }
                            Ok(n) => total_bytes_read += n,
                            Err(e) => return Some(Err(e.into())),
                        }
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use bytes::BytesMut;
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_panic;

    use super::*;
    use crate::messages::DecoderMessage;
    use crate::messages::EncoderMessage;
    use crate::messages::MessageEncoder;
    use crate::test::SESSION;

    /// A `Read` that yields a single byte per call, forcing maximal partial reads.
    struct ByteAtATimeReader {
        data: Vec<u8>,
        pos: usize,
    }

    impl Read for ByteAtATimeReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if buf.is_empty() || self.pos >= self.data.len() {
                return Ok(0);
            }
            buf[0] = self.data[self.pos];
            self.pos += 1;
            Ok(1)
        }
    }

    #[test]
    fn reads_message_split_across_short_reads() -> VortexResult<()> {
        let array = buffer![0i32, 1, 2, 3].into_array();
        let mut ipc_bytes = BytesMut::new();
        let mut encoder = MessageEncoder::new(SESSION.clone());
        for buf in encoder.encode(EncoderMessage::Array(&array))? {
            ipc_bytes.extend_from_slice(buf.as_ref());
        }

        let reader = ByteAtATimeReader {
            data: ipc_bytes.to_vec(),
            pos: 0,
        };
        let mut messages = SyncMessageReader::new(reader);

        let DecoderMessage::Array((parts, ctx, row_count)) =
            messages.next().vortex_expect("expected a message")?
        else {
            vortex_panic!("expected an array message");
        };
        let actual = parts.decode(array.dtype(), row_count, &ctx, &SESSION)?;
        assert_eq!(actual.len(), array.len());
        assert_eq!(actual.encoding_id(), array.encoding_id());
        assert!(messages.next().is_none());
        Ok(())
    }
}

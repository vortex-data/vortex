// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Read;

use bytes::BytesMut;
use vortex_error::VortexResult;

use crate::messages::DecoderMessage;
use crate::messages::MessageDecoder;
use crate::messages::PollRead;

#[derive(Default)]
enum ReadState {
    #[default]
    AwaitingDecoder,
    Filling {
        total_bytes_read: usize,
    },
    Done,
}

/// An IPC message reader backed by a `Read` stream.
pub struct SyncMessageReader<R> {
    read: R,
    buffer: BytesMut,
    decoder: MessageDecoder,
    state: ReadState,
}

impl<R: Read> SyncMessageReader<R> {
    pub fn new(read: R) -> Self {
        SyncMessageReader {
            read,
            buffer: BytesMut::new(),
            decoder: MessageDecoder::default(),
            state: ReadState::default(),
        }
    }
}

impl<R: Read> Iterator for SyncMessageReader<R> {
    type Item = VortexResult<DecoderMessage>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match &mut self.state {
                ReadState::Done => return None,
                ReadState::AwaitingDecoder => match self.decoder.read_next(&mut self.buffer) {
                    Ok(PollRead::Some(msg)) => {
                        return Some(Ok(msg));
                    }
                    Ok(PollRead::NeedMore(nbytes)) => {
                        self.buffer.resize(nbytes, 0x00);
                        self.state = ReadState::Filling {
                            total_bytes_read: 0,
                        };
                    }
                    Err(e) => {
                        self.state = ReadState::Done;
                        return Some(Err(e));
                    }
                },
                ReadState::Filling { total_bytes_read } => {
                    // `Read` can return fewer bytes than requested; fill completely
                    // before decoding, which reads `nbytes` regardless of how many
                    // were actually written.
                    while *total_bytes_read < self.buffer.len() {
                        match self.read.read(&mut self.buffer[*total_bytes_read..]) {
                            Ok(0) => {
                                let bytes_read = *total_bytes_read;
                                self.buffer.clear();
                                self.state = ReadState::Done;
                                return match self.decoder.finish_eof(bytes_read) {
                                    Ok(()) => None,
                                    Err(e) => Some(Err(e)),
                                };
                            }
                            Ok(n) => *total_bytes_read += n,
                            Err(e) => return Some(Err(e.into())),
                        }
                    }
                    self.state = ReadState::AwaitingDecoder;
                }
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
    use crate::test::assert_io_error_kind;

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

    #[test]
    fn truncated_message_errors_once_then_terminates() -> VortexResult<()> {
        let array = buffer![0i32, 1, 2, 3].into_array();
        let mut encoder = MessageEncoder::new(SESSION.clone());
        let parts = encoder.encode(EncoderMessage::Array(&array))?;
        let header_len = parts[1].len();
        let body_len: usize = parts[2..].iter().map(|b| b.len()).sum();
        let full: Vec<u8> = parts.iter().flat_map(|b| b.as_ref()).copied().collect();

        // Cut at: partial length, exact after-length, partial header, exact after-header, partial body.
        let cut_points = [
            2,
            4,
            4 + header_len / 2,
            4 + header_len,
            4 + header_len + body_len / 2,
        ];

        for cut in cut_points {
            let reader = ByteAtATimeReader {
                data: full[..cut].to_vec(),
                pos: 0,
            };
            let mut messages = SyncMessageReader::new(reader);
            let err = messages
                .next()
                .vortex_expect("expected error on truncated frame")
                .unwrap_err();
            assert_io_error_kind(&err, std::io::ErrorKind::UnexpectedEof);
            assert!(messages.next().is_none());
            assert!(messages.next().is_none());
        }

        Ok(())
    }

    #[test]
    fn transport_error_is_distinct_and_preserves_progress() -> VortexResult<()> {
        struct FlakyReader {
            data: Vec<u8>,
            pos: usize,
            errored: bool,
        }

        impl Read for FlakyReader {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if !self.errored && self.pos == 1 {
                    self.errored = true;
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "transient read error",
                    ));
                }
                if buf.is_empty() || self.pos >= self.data.len() {
                    return Ok(0);
                }
                buf[0] = self.data[self.pos];
                self.pos += 1;
                Ok(1)
            }
        }

        let array = buffer![7i32, 8, 9].into_array();
        let mut encoder = MessageEncoder::new(SESSION.clone());
        let full: Vec<u8> = encoder
            .encode(EncoderMessage::Array(&array))?
            .iter()
            .flat_map(|b| b.as_ref())
            .copied()
            .collect();

        let mut messages = SyncMessageReader::new(FlakyReader {
            data: full,
            pos: 0,
            errored: false,
        });

        let err = messages
            .next()
            .vortex_expect("expected transport error")
            .unwrap_err();
        assert_io_error_kind(&err, std::io::ErrorKind::Interrupted);

        let DecoderMessage::Array((parts, ctx, row_count)) = messages
            .next()
            .vortex_expect("expected recovered message")?
        else {
            vortex_panic!("expected an array message");
        };
        let actual = parts.decode(array.dtype(), row_count, &ctx, &SESSION)?;
        assert_eq!(actual.len(), array.len());
        assert!(messages.next().is_none());
        Ok(())
    }
}

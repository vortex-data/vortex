// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::io::Read;

use bytes::BytesMut;
use vortex_error::VortexResult;

use crate::messages::DecoderMessage;
use crate::messages::MessageDecoder;
use crate::messages::PollRead;

/// Maximum number of bytes to grow the read buffer by before confirming the stream provides data.
pub(super) const MAX_READ_CHUNK: usize = 64 * 1024;

/// Grows `buffer`, which holds `filled` bytes of a `target_len`-byte frame, for the next read.
///
/// A new allocation is at most twice the bytes already received, or `filled + MAX_READ_CHUNK`,
/// so an untrusted length prefix cannot pre-allocate memory. Growth is geometric, so a large frame
/// costs O(n) copies, and the last allocation is exactly `target_len` bytes.
pub(super) fn grow_frame_buffer(buffer: &mut BytesMut, filled: usize, target_len: usize) {
    let next_len = filled.saturating_add(MAX_READ_CHUNK).min(target_len);
    if next_len > buffer.capacity() {
        let new_cap = filled.saturating_mul(2).max(next_len).min(target_len);
        let mut grown = BytesMut::with_capacity(new_cap);
        grown.extend_from_slice(&buffer[..filled]);
        *buffer = grown;
    }
    buffer.resize(next_len, 0x00);
}

#[derive(Default)]
enum ReadState {
    #[default]
    AwaitingDecoder,
    Filling {
        target_len: usize,
        total_bytes_read: usize,
    },
    Done,
}

/// An IPC message reader backed by a `Read` stream.
///
/// # Error handling
///
/// - [`io::ErrorKind::Interrupted`] is retried transparently in place.
/// - [`io::ErrorKind::WouldBlock`] and [`io::ErrorKind::TimedOut`] are returned to the caller
///   while preserving any partial frame already buffered so the next call resumes where reading
///   stopped.
/// - Any other I/O error, EOF error, or decode error clears the internal buffer and terminates
///   the reader so subsequent calls return `None`.
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
                        self.buffer.clear();
                        self.state = ReadState::Filling {
                            target_len: nbytes,
                            total_bytes_read: 0,
                        };
                    }
                    Err(e) => {
                        self.buffer.clear();
                        self.state = ReadState::Done;
                        return Some(Err(e));
                    }
                },
                ReadState::Filling {
                    target_len,
                    total_bytes_read,
                } => {
                    // `Read` can return fewer bytes than requested; fill completely
                    // before decoding, which reads `nbytes` regardless of how many
                    // were actually written. Grow in bounded chunks so an untrusted
                    // length prefix on a short stream cannot pre-allocate gigabytes.
                    while *total_bytes_read < *target_len {
                        if self.buffer.len() == *total_bytes_read {
                            grow_frame_buffer(&mut self.buffer, *total_bytes_read, *target_len);
                        }
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
                            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                            Err(e)
                                if matches!(
                                    e.kind(),
                                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                                ) =>
                            {
                                return Some(Err(e.into()));
                            }
                            Err(e) => {
                                self.buffer.clear();
                                self.state = ReadState::Done;
                                return Some(Err(e.into()));
                            }
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
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
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
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
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
            assert_io_error_kind(&err, io::ErrorKind::UnexpectedEof);
            assert!(messages.next().is_none());
            assert!(messages.next().is_none());
        }

        Ok(())
    }

    #[test]
    fn test_huge_length_prefix_on_short_stream_returns_eof_without_oom() {
        let mut messages =
            SyncMessageReader::new(io::Cursor::new([0xff, 0xff, 0xff, 0xff, 0x00, 0x01]));
        let err = messages
            .next()
            .vortex_expect("expected error on truncated frame")
            .unwrap_err();
        assert_io_error_kind(&err, io::ErrorKind::UnexpectedEof);
        assert!(messages.next().is_none());
    }

    #[test]
    fn test_multi_chunk_frame_decodes_cleanly() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter(0i32..20_000).into_array();
        let mut encoder = MessageEncoder::new(SESSION.clone());
        let full: Vec<u8> = encoder
            .encode(EncoderMessage::Array(&array))?
            .iter()
            .flat_map(|b| b.as_ref())
            .copied()
            .collect();
        assert!(full.len() > MAX_READ_CHUNK);

        let mut messages = SyncMessageReader::new(io::Cursor::new(full));
        let DecoderMessage::Array((parts, ctx, row_count)) = messages
            .next()
            .vortex_expect("expected multi-chunk array message")?
        else {
            vortex_panic!("expected an array message");
        };
        let actual = parts.decode(array.dtype(), row_count, &ctx, &SESSION)?;
        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(actual, array, &mut exec_ctx);
        assert!(messages.next().is_none());
        Ok(())
    }

    #[test]
    fn test_grow_frame_buffer_bounds_and_exact_final_capacity() {
        let target_len = 1_000_003;
        let mut buffer = BytesMut::new();
        let mut filled = 0;

        while filled < target_len {
            grow_frame_buffer(&mut buffer, filled, target_len);
            let max_allowed = filled
                .saturating_mul(2)
                .max(filled.saturating_add(MAX_READ_CHUNK));
            assert!(buffer.capacity() <= max_allowed);
            filled = buffer.len();
        }

        assert_eq!(buffer.len(), target_len);
        assert_eq!(buffer.capacity(), target_len);
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
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
                        self.buffer.clear();
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
                            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
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
    use std::io;
    use std::io::Read;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

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
    use crate::messages::DecoderMessage;
    use crate::messages::EncoderMessage;
    use crate::messages::MessageEncoder;
    use crate::tests::SESSION;
    use crate::tests::assert_io_error_kind;

    /// A `Read` that yields at most `chunk_size` bytes per call.
    struct ChunkedReader {
        data: Vec<u8>,
        pos: usize,
        chunk_size: usize,
    }

    impl ChunkedReader {
        fn new(data: Vec<u8>, chunk_size: usize) -> Self {
            Self {
                data,
                pos: 0,
                chunk_size,
            }
        }
    }

    impl Read for ChunkedReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if buf.is_empty() || self.pos >= self.data.len() {
                return Ok(0);
            }
            let remaining = self.data.len() - self.pos;
            let count = buf.len().min(self.chunk_size).min(remaining);
            buf[..count].copy_from_slice(&self.data[self.pos..self.pos + count]);
            self.pos += count;
            Ok(count)
        }
    }

    #[test]
    fn reads_messages_split_across_small_chunk_sizes() -> VortexResult<()> {
        let array = buffer![0i32, 1, 2, 3, 4, 5].into_array();
        let raw_buf = ByteBuffer::copy_from_aligned([10u8, 20, 30, 40, 50], Alignment::new(64));
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
            let mut messages =
                SyncMessageReader::new(ChunkedReader::new(encoded.clone(), chunk_size));

            let DecoderMessage::DType(fb_dtype) =
                messages.next().vortex_expect("expected dtype message")?
            else {
                vortex_panic!("expected a dtype message");
            };
            assert_eq!(DType::from_flatbuffer(fb_dtype, &SESSION)?, *array.dtype());

            let DecoderMessage::Array((parts, ctx, row_count)) =
                messages.next().vortex_expect("expected array message")?
            else {
                vortex_panic!("expected an array message");
            };
            let actual = parts.decode(array.dtype(), row_count, &ctx, &SESSION)?;
            assert_arrays_eq!(actual, array, &mut SESSION.create_execution_ctx());

            let DecoderMessage::Buffer(decoded_buf) =
                messages.next().vortex_expect("expected buffer message")?
            else {
                vortex_panic!("expected a buffer message");
            };
            assert_eq!(decoded_buf.as_slice(), raw_buf.as_slice());
            assert_eq!(decoded_buf.alignment(), Alignment::new(64));
            assert!(Alignment::new(64).is_ptr_aligned(decoded_buf.as_ptr()));

            assert!(messages.next().is_none());
        }

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
            let mut messages = SyncMessageReader::new(ChunkedReader::new(full[..cut].to_vec(), 1));
            let Err(err) = messages.next().vortex_expect("truncated frame") else {
                vortex_panic!("expected error on truncated frame");
            };
            assert_io_error_kind(&err, io::ErrorKind::UnexpectedEof);
            assert!(messages.next().is_none());
            assert!(messages.next().is_none());
        }

        Ok(())
    }

    #[test]
    fn interrupted_read_retries_and_preserves_progress() -> VortexResult<()> {
        struct InterruptedReader {
            data: Vec<u8>,
            pos: usize,
            interrupt_positions: Vec<usize>,
        }

        impl Read for InterruptedReader {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if let Some(idx) = self.interrupt_positions.iter().position(|p| *p == self.pos) {
                    self.interrupt_positions.remove(idx);
                    return Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "transient interrupt",
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
        let parts = encoder.encode(EncoderMessage::Array(&array))?;
        let header_len = parts[1].len();
        let full: Vec<u8> = parts.iter().flat_map(|b| b.as_ref()).copied().collect();

        let mut messages = SyncMessageReader::new(InterruptedReader {
            data: full,
            pos: 0,
            interrupt_positions: vec![1, 4 + header_len / 2, 4 + header_len + 1],
        });

        let DecoderMessage::Array((parts, ctx, row_count)) =
            messages.next().vortex_expect("recovered message")?
        else {
            vortex_panic!("expected an array message");
        };
        let actual = parts.decode(array.dtype(), row_count, &ctx, &SESSION)?;
        assert_arrays_eq!(actual, array, &mut SESSION.create_execution_ctx());
        assert!(messages.next().is_none());
        Ok(())
    }

    #[test]
    fn io_error_during_fill_is_terminal() -> VortexResult<()> {
        struct FailingReader {
            data: Vec<u8>,
            pos: usize,
            fail_at: usize,
            read_calls: Arc<AtomicUsize>,
        }

        impl Read for FailingReader {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                self.read_calls.fetch_add(1, Ordering::SeqCst);
                if self.pos >= self.fail_at {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "broken transport",
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
        let parts = encoder.encode(EncoderMessage::Array(&array))?;
        let header_len = parts[1].len();
        let full: Vec<u8> = parts.iter().flat_map(|b| b.as_ref()).copied().collect();

        for fail_at in [1, 4 + header_len / 2, 4 + header_len + 1] {
            let read_calls = Arc::new(AtomicUsize::new(0));
            let mut messages = SyncMessageReader::new(FailingReader {
                data: full.clone(),
                pos: 0,
                fail_at,
                read_calls: Arc::clone(&read_calls),
            });

            let Err(err) = messages.next().vortex_expect("expected io error") else {
                vortex_panic!("expected io error");
            };
            assert_io_error_kind(&err, io::ErrorKind::BrokenPipe);

            let calls_after_err = read_calls.load(Ordering::SeqCst);
            assert!(messages.next().is_none());
            assert!(messages.next().is_none());
            assert_eq!(read_calls.load(Ordering::SeqCst), calls_after_err);
        }

        Ok(())
    }
}

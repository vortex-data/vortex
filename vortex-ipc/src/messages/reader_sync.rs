use std::io::Read;

use bytes::BytesMut;
use vortex_array::ArrayRegistry;
use vortex_error::VortexResult;

use crate::messages::{DecoderMessage, MessageDecoder, PollRead};

/// An IPC message reader backed by a `Read` stream.
pub struct SyncMessageReader<R> {
    read: R,
    buffer: BytesMut,
    decoder: MessageDecoder,
}

impl<R: Read> SyncMessageReader<R> {
    pub fn new(read: R, registry: ArrayRegistry) -> Self {
        SyncMessageReader {
            read,
            buffer: BytesMut::new(),
            decoder: MessageDecoder::new(registry),
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
                    match self.read.read(&mut self.buffer) {
                        Ok(0) => {
                            // EOF
                            return None;
                        }
                        Ok(_nbytes) => {
                            // Continue in the loop
                        }
                        Err(e) => return Some(Err(e.into())),
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

use std::io::Read;

use bytes::BytesMut;
use vortex_error::VortexResult;

use crate::messages::{DecoderMessage, MessageDecoder, NextMessage};

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

    pub fn read_message(&mut self) -> VortexResult<Option<DecoderMessage>> {
        loop {
            match self.decoder.read_next(&mut self.buffer)? {
                NextMessage::Some(msg) => {
                    return Ok(Some(msg));
                }
                NextMessage::NeedMore(nbytes) => {
                    self.buffer.resize(nbytes, 0x00);
                    match self.read.read(&mut self.buffer) {
                        Ok(0) => {
                            // EOF
                            return Ok(None);
                        }
                        Ok(_nbytes) => {
                            // Continue in the loop
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
            }
        }
    }
}

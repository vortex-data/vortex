use std::io::Write;

use vortex_error::VortexResult;

use crate::messages::{EncoderMessage, MessageEncoder};

pub struct SyncMessageWriter<W> {
    write: W,
    encoder: MessageEncoder,
}

impl<W: Write> SyncMessageWriter<W> {
    pub fn new(write: W) -> Self {
        Self {
            write,
            encoder: MessageEncoder::default(),
        }
    }

    pub fn write_message(&mut self, message: EncoderMessage) -> VortexResult<()> {
        for buffer in self.encoder.encode(message) {
            self.write.write_all(&buffer)?;
        }
        Ok(())
    }
}

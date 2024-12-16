use futures_util::{AsyncWrite, AsyncWriteExt};
use vortex_error::VortexResult;

use crate::messages::{EncoderMessage, MessageEncoder};
use crate::ALIGNMENT;

pub struct AsyncMessageWriter<W> {
    write: W,
    encoder: MessageEncoder,
}

impl<W: AsyncWrite + Unpin> AsyncMessageWriter<W> {
    pub fn new(write: W) -> Self {
        Self {
            write,
            encoder: MessageEncoder::new(ALIGNMENT),
        }
    }

    pub async fn write_message(&mut self, message: EncoderMessage<'_>) -> VortexResult<()> {
        for buffer in self.encoder.encode(message) {
            self.write.write_all(&buffer).await?;
        }
        Ok(())
    }
}

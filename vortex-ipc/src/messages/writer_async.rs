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

    pub fn get_mut(&mut self) -> &mut W {
        &mut self.write
    }

    pub fn into_inner(self) -> W {
        self.write
    }
}

impl<W> AsRef<W> for AsyncMessageWriter<W> {
    fn as_ref(&self) -> &W {
        &self.write
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::AsyncWrite;
use futures::AsyncWriteExt;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::messages::EncoderMessage;
use crate::messages::MessageEncoder;

pub struct AsyncMessageWriter<W> {
    write: W,
    encoder: MessageEncoder,
}

impl<W: AsyncWrite + Unpin> AsyncMessageWriter<W> {
    pub fn new(write: W, session: &VortexSession) -> Self {
        Self {
            write,
            encoder: MessageEncoder::new(session.clone()),
        }
    }

    pub async fn write_message(&mut self, message: EncoderMessage<'_>) -> VortexResult<()> {
        for buffer in self.encoder.encode(message)? {
            self.write.write_all(&buffer).await?;
        }
        Ok(())
    }

    pub fn inner(&self) -> &W {
        &self.write
    }

    pub fn into_inner(self) -> W {
        self.write
    }
}

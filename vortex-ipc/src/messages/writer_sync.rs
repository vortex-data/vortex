// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::messages::EncoderMessage;
use crate::messages::MessageEncoder;

pub struct SyncMessageWriter<W> {
    write: W,
    encoder: MessageEncoder,
}

impl<W: Write> SyncMessageWriter<W> {
    pub fn new(write: W, session: &VortexSession) -> Self {
        Self {
            write,
            encoder: MessageEncoder::new(session.clone()),
        }
    }

    pub fn write_message(&mut self, message: EncoderMessage) -> VortexResult<()> {
        for buffer in self.encoder.encode(message)? {
            self.write.write_all(&buffer)?;
        }
        Ok(())
    }
}

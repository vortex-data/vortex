// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bytes::Buf;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::messages::DecoderMessage;
use crate::messages::MessageDecoder;
use crate::messages::MessageLimits;
use crate::messages::PollRead;

/// An IPC message reader backed by a `Read` stream.
pub struct BufMessageReader<B> {
    buffer: B,
    decoder: MessageDecoder,
}

impl<B: Buf> BufMessageReader<B> {
    pub fn new(buffer: B) -> Self {
        Self::with_limits(buffer, MessageLimits::default())
    }

    /// Create a reader that enforces the given [`MessageLimits`] on the buffered messages.
    pub fn with_limits(buffer: B, limits: MessageLimits) -> Self {
        BufMessageReader {
            buffer,
            decoder: MessageDecoder::new(limits),
        }
    }
}

impl<B: Buf> Iterator for BufMessageReader<B> {
    type Item = VortexResult<DecoderMessage>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.buffer.has_remaining() {
            // End-of-buffer reached
            return None;
        }
        match self.decoder.read_next(&mut self.buffer) {
            Ok(PollRead::Some(msg)) => Some(Ok(msg)),
            Ok(PollRead::NeedMore(_)) => Some(Err(vortex_err!(
                "Buffer did not have sufficient bytes for an IPC message"
            ))),
            Err(e) => Some(Err(e)),
        }
    }
}

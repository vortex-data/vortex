use bytes::Buf;
use vortex_error::{vortex_err, VortexResult};

use crate::messages::{DecoderMessage, MessageDecoder, PollRead};

/// An IPC message reader backed by a `Read` stream.
pub struct BufMessageReader<B> {
    buffer: B,
    decoder: MessageDecoder,
}

impl<B: Buf> BufMessageReader<B> {
    pub fn new(buffer: B) -> Self {
        BufMessageReader {
            buffer,
            decoder: MessageDecoder::default(),
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

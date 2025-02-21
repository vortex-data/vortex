use std::fmt::Debug;

use bytes::Buf;
use flatbuffers::{root, root_unchecked};
use vortex_array::serde::ArrayParts;
use vortex_buffer::{AlignedBuf, Alignment, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_flatbuffers::message::{MessageHeader, MessageVersion};
use vortex_flatbuffers::{dtype as fbd, message as fb, FlatBuffer};

/// A message decoded from an IPC stream.
///
/// Note that the `Array` variant cannot fully decode into an [`vortex_array::Array`] without
/// a [`vortex_array::ContextRef`] and a [`DType`]. As such, we partially decode into an
/// [`ArrayParts`] and allow the caller to finish the decoding.
#[derive(Debug)]
pub enum DecoderMessage {
    Array((ArrayParts, usize)),
    Buffer(ByteBuffer),
    DType(DType),
}

#[derive(Default)]
enum State {
    #[default]
    Length,
    Header(usize),
    Reading(FlatBuffer),
}

#[derive(Debug)]
pub enum PollRead {
    Some(DecoderMessage),
    /// Returns the _total_ number of bytes needed to make progress.
    /// Note this is _not_ the incremental number of bytes needed to make progress.
    NeedMore(usize),
}

// NOTE(ngates): we should design some trait that the Decoder can take that doesn't require unique
//  ownership of the underlying bytes. The decoder needs to split out bytes, and advance a cursor,
//  but it doesn't need to mutate any bytes. So in theory, we should be able to do this zero-copy
//  over a shared buffer of bytes, instead of requiring a `BytesMut`.
/// A stateful reader for decoding IPC messages from an arbitrary stream of bytes.
#[derive(Default)]
pub struct MessageDecoder {
    /// The current state of the decoder.
    state: State,
}

impl MessageDecoder {
    /// Attempt to read the next message from the bytes object.
    ///
    /// If the message is incomplete, the function will return `NeedMore` with the _total_ number
    /// of bytes needed to make progress. The next call to read_next _should_ provide at least
    /// this number of bytes otherwise it will be given the same `NeedMore` response.
    pub fn read_next<B: AlignedBuf>(&mut self, bytes: &mut B) -> VortexResult<PollRead> {
        loop {
            match &self.state {
                State::Length => {
                    if bytes.remaining() < 4 {
                        return Ok(PollRead::NeedMore(4));
                    }

                    let msg_length = bytes.get_u32_le();
                    self.state = State::Header(msg_length as usize);
                }
                State::Header(msg_length) => {
                    if bytes.remaining() < *msg_length {
                        return Ok(PollRead::NeedMore(*msg_length));
                    }

                    let msg_bytes = bytes.copy_to_const_aligned(*msg_length);
                    let msg = root::<fb::Message>(msg_bytes.as_ref())?;
                    if msg.version() != MessageVersion::V0 {
                        vortex_bail!("Unsupported message version {:?}", msg.version());
                    }

                    self.state = State::Reading(msg_bytes);
                }
                State::Reading(msg_bytes) => {
                    // SAFETY: we've already validated the header in the previous state
                    let msg = unsafe { root_unchecked::<fb::Message>(msg_bytes.as_ref()) };

                    // Now we read the body
                    let body_length = usize::try_from(msg.body_size()).map_err(|_| {
                        vortex_err!("body size {} is too large for usize", msg.body_size())
                    })?;
                    if bytes.remaining() < body_length {
                        return Ok(PollRead::NeedMore(body_length));
                    }

                    match msg.header_type() {
                        MessageHeader::ArrayMessage => {
                            // We don't care about alignment here since ArrayParts will handle it.
                            let body = bytes.copy_to_aligned(body_length, Alignment::new(1));
                            let parts = ArrayParts::try_from(body)?;

                            let row_count = msg
                                .header_as_array_message()
                                .vortex_expect("header is array")
                                .row_count() as usize;

                            self.state = Default::default();
                            return Ok(PollRead::Some(DecoderMessage::Array((parts, row_count))));
                        }
                        MessageHeader::BufferMessage => {
                            let body = bytes.copy_to_aligned(
                                body_length,
                                Alignment::from_exponent(
                                    msg.header_as_buffer_message()
                                        .vortex_expect("header is buffer")
                                        .alignment_exponent(),
                                ),
                            );

                            self.state = Default::default();
                            return Ok(PollRead::Some(DecoderMessage::Buffer(body)));
                        }
                        MessageHeader::DTypeMessage => {
                            let body: FlatBuffer = bytes.copy_to_const_aligned::<8>(body_length);
                            let fb_dtype = root::<fbd::DType>(body.as_ref())?;
                            let dtype = DType::try_from_view(fb_dtype, body.clone())?;

                            self.state = Default::default();
                            return Ok(PollRead::Some(DecoderMessage::DType(dtype)));
                        }
                        _ => {
                            vortex_bail!("Unsupported message header {:?}", msg.header_type());
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use bytes::BytesMut;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::{Array, IntoArray};
    use vortex_buffer::buffer;
    use vortex_error::vortex_panic;

    use super::*;
    use crate::messages::{EncoderMessage, MessageEncoder};

    fn write_and_read(expected: Array) {
        let mut ipc_bytes = BytesMut::new();
        let mut encoder = MessageEncoder::default();
        for buf in encoder.encode(EncoderMessage::Array(&expected)) {
            ipc_bytes.extend_from_slice(buf.as_ref());
        }

        let mut decoder = MessageDecoder::default();

        // Since we provide all bytes up-front, we should never hit a NeedMore.
        let mut buffer = BytesMut::from(ipc_bytes.as_ref());
        let (array_parts, row_count) = match decoder.read_next(&mut buffer).unwrap() {
            PollRead::Some(DecoderMessage::Array(array_parts)) => array_parts,
            otherwise => vortex_panic!("Expected an array, got {:?}", otherwise),
        };

        // Decode the array parts with the context
        let actual = array_parts
            .decode(Default::default(), expected.dtype().clone(), row_count)
            .unwrap();

        assert_eq!(expected.len(), actual.len());
        assert_eq!(expected.encoding(), actual.encoding());
    }

    #[test]
    fn array_ipc() {
        write_and_read(buffer![0i32, 1, 2, 3].into_array());
    }

    #[test]
    fn array_no_buffers() {
        // Constant arrays have a single buffer
        let array = ConstantArray::new(10i32, 20).into_array();
        assert_eq!(array.nbuffers(), 1, "Array should have a single buffer");
        write_and_read(array);
    }
}

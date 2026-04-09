// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use bytes::Buf;
use flatbuffers::root;
use flatbuffers::root_unchecked;
use vortex_array::ArrayId;
use vortex_array::serde::SerializedArray;
use vortex_buffer::AlignedBuf;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::message as fb;
use vortex_flatbuffers::message::MessageHeader;
use vortex_flatbuffers::message::MessageVersion;
use vortex_session::registry::ReadContext;

/// A message decoded from an IPC stream.
#[derive(Debug)]
pub enum DecoderMessage {
    Array((SerializedArray, ReadContext, usize)),
    Buffer(ByteBuffer),
    DType(FlatBuffer),
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
    /// A complete message was decoded.
    Some(DecoderMessage),
    /// The decoder needs more data to make progress.
    ///
    /// The inner value is the **total*k number of bytes the buffer should contain, not the
    /// incremental amount needed. Callers should:
    ///
    /// 1. Resize the buffer to this length.
    /// 2. Fill the buffer completely (handling partial reads as needed).
    /// 3. Only then call [`MessageDecoder::read_next`] again.
    ///
    /// The decoder checks [`bytes::Buf::remaining`] to determine available data, which for
    /// [`bytes::BytesMut`] returns the buffer length regardless of how many bytes were actually
    /// written. Calling `read_next` before the buffer is fully populated will cause the decoder
    /// to read garbage data.
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
                            let parts = SerializedArray::try_from(body)?;

                            let header = msg
                                .header_as_array_message()
                                .vortex_expect("header is array");

                            let encoding_ids: Arc<_> = header
                                .encodings()
                                .iter()
                                .flat_map(|e| e.iter())
                                .map(|id| ArrayId::new_arc(Arc::from(id.to_string())))
                                .collect();

                            let ctx = ReadContext::new(encoding_ids);
                            let row_count = header.row_count() as usize;

                            self.state = Default::default();
                            return Ok(PollRead::Some(DecoderMessage::Array((
                                parts, ctx, row_count,
                            ))));
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
                            let dtype: FlatBuffer = bytes.copy_to_const_aligned::<8>(body_length);
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
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_buffer::buffer;
    use vortex_error::vortex_panic;

    use super::*;
    use crate::messages::EncoderMessage;
    use crate::messages::MessageEncoder;
    use crate::test::SESSION;

    fn write_and_read(expected: &ArrayRef) {
        let mut ipc_bytes = BytesMut::new();
        let mut encoder = MessageEncoder::new(SESSION.clone());
        for buf in encoder.encode(EncoderMessage::Array(expected)).unwrap() {
            ipc_bytes.extend_from_slice(buf.as_ref());
        }

        let mut decoder = MessageDecoder::default();

        // Since we provide all bytes up-front, we should never hit a NeedMore.
        let mut buffer = BytesMut::from(ipc_bytes.as_ref());
        let (array_parts, ctx, row_count) = match decoder.read_next(&mut buffer).unwrap() {
            PollRead::Some(DecoderMessage::Array(array_parts)) => array_parts,
            otherwise => vortex_panic!("Expected an array, got {:?}", otherwise),
        };

        // Decode the array parts with the context
        let actual = array_parts
            .decode(expected.dtype(), row_count, &ctx, &SESSION)
            .unwrap();

        assert_eq!(expected.len(), actual.len());
        assert_eq!(expected.encoding_id(), actual.encoding_id());
    }

    #[test]
    fn array_ipc() {
        write_and_read(&buffer![0i32, 1, 2, 3].into_array());
    }

    #[test]
    fn array_no_buffers() {
        // Constant arrays have a single buffer
        let array = ConstantArray::new(10i32, 20);
        assert_eq!(array.nbuffers(), 1, "Array should have a single buffer");
        write_and_read(&array.into_array());
    }
}

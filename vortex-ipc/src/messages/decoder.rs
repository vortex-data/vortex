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

/// Upper bounds on the sizes an IPC message is allowed to declare.
///
/// A message declares the length of its flatbuffer header and of its body before either has been
/// received. Callers of [`MessageDecoder::read_next`] are expected to allocate a buffer of the
/// size reported by [`PollRead::NeedMore`], so an unbounded declared size lets a few bytes of
/// input request an arbitrarily large allocation. The decoder rejects a message whose declared
/// sizes exceed these limits before reporting `NeedMore`, so no caller ever allocates on behalf
/// of an oversized declaration.
///
/// The defaults are far above any message Vortex writes; raise them only when a producer is
/// known to emit larger messages, and prefer [`Self::UNLIMITED`] only for trusted local input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageLimits {
    /// The largest flatbuffer header, in bytes, that a message may declare.
    pub max_header_size: usize,
    /// The largest body, in bytes, that a message may declare.
    pub max_body_size: usize,
}

impl MessageLimits {
    /// The default header limit: 16 MiB.
    ///
    /// A header holds only flatbuffer-encoded metadata (the encoding ids, row count, and dtype),
    /// which stays in the kilobytes even for very wide schemas.
    pub const DEFAULT_MAX_HEADER_SIZE: usize = 16 << 20;

    /// The default body limit: 1 GiB.
    ///
    /// A body holds the serialized buffers of a single array message. Writers chunk their output
    /// well below this, so the limit only rejects declarations that could not have come from a
    /// well-formed stream.
    pub const DEFAULT_MAX_BODY_SIZE: usize = 1 << 30;

    /// Limits that permit any size representable by the format.
    ///
    /// Only use this when the byte stream is trusted: it restores the behaviour where a malformed
    /// message can request an arbitrarily large allocation.
    pub const UNLIMITED: Self = Self {
        max_header_size: usize::MAX,
        max_body_size: usize::MAX,
    };
}

impl Default for MessageLimits {
    fn default() -> Self {
        Self {
            max_header_size: Self::DEFAULT_MAX_HEADER_SIZE,
            max_body_size: Self::DEFAULT_MAX_BODY_SIZE,
        }
    }
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
    /// Bounds on the sizes a message may declare.
    limits: MessageLimits,
}

impl MessageDecoder {
    /// Create a decoder that enforces the given [`MessageLimits`].
    pub fn new(limits: MessageLimits) -> Self {
        Self {
            state: State::default(),
            limits,
        }
    }

    /// The size limits this decoder enforces.
    pub fn limits(&self) -> MessageLimits {
        self.limits
    }

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

                    let msg_length = bytes.get_u32_le() as usize;
                    // Checked before `NeedMore` so that a caller sizing its buffer from the
                    // reported value never allocates for an oversized declaration.
                    if msg_length > self.limits.max_header_size {
                        vortex_bail!(
                            "IPC message header size {msg_length} exceeds the limit of {} bytes",
                            self.limits.max_header_size
                        );
                    }
                    self.state = State::Header(msg_length);
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
                    // As above: reject an oversized declaration before a caller sizes its buffer
                    // from the `NeedMore` below.
                    if body_length > self.limits.max_body_size {
                        vortex_bail!(
                            "IPC message body size {body_length} exceeds the limit of {} bytes",
                            self.limits.max_body_size
                        );
                    }
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

                            #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
                            let encoding_ids: Arc<_> = header
                                .encodings()
                                .iter()
                                .flat_map(|e| e.iter())
                                .map(ArrayId::new)
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
                                Alignment::try_from_untrusted_exponent(
                                    msg.header_as_buffer_message()
                                        .vortex_expect("header is buffer")
                                        .alignment_exponent(),
                                )?,
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

    /// Build a `BufferMessage` header declaring `body_size` bytes, prefixed by its length.
    ///
    /// The body itself is omitted: the point is that the decoder must reject the declaration
    /// before any caller sizes a buffer from it.
    fn message_declaring_body(body_size: u64) -> BytesMut {
        let mut fbb = flatbuffers::FlatBufferBuilder::new();
        let header = fb::BufferMessage::create(
            &mut fbb,
            &fb::BufferMessageArgs {
                alignment_exponent: 0,
            },
        )
        .as_union_value();

        let mut msg = fb::MessageBuilder::new(&mut fbb);
        msg.add_version(Default::default());
        msg.add_header_type(MessageHeader::BufferMessage);
        msg.add_header(header);
        msg.add_body_size(body_size);
        let msg = msg.finish();
        fbb.finish_minimal(msg);

        let header_bytes = fbb.finished_data();
        let mut out = BytesMut::new();
        out.extend_from_slice(&u32::try_from(header_bytes.len()).unwrap().to_le_bytes());
        out.extend_from_slice(header_bytes);
        out
    }

    #[test]
    fn rejects_oversized_declared_header() {
        // Four bytes of input claiming a ~4 GiB header.
        let mut buffer = BytesMut::from(&u32::MAX.to_le_bytes()[..]);

        let err = MessageDecoder::default()
            .read_next(&mut buffer)
            .expect_err("an oversized header declaration must be rejected");
        assert!(
            err.to_string().contains("exceeds the limit"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_oversized_declared_body() {
        // A well-formed header declaring a 256 TiB body, with none of it present.
        let mut buffer = message_declaring_body(1 << 48);

        let err = MessageDecoder::default()
            .read_next(&mut buffer)
            .expect_err("an oversized body declaration must be rejected");
        assert!(
            err.to_string().contains("exceeds the limit"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn body_within_limit_asks_for_more_bytes() {
        // A body under the limit still round-trips through `NeedMore`, so the check rejects only
        // oversized declarations rather than short-circuiting the normal incomplete-read path.
        let mut buffer = message_declaring_body(4096);

        match MessageDecoder::default().read_next(&mut buffer).unwrap() {
            PollRead::NeedMore(n) => assert_eq!(n, 4096),
            otherwise => vortex_panic!("Expected NeedMore, got {:?}", otherwise),
        }
    }

    #[test]
    fn limits_are_configurable() {
        let limits = MessageLimits {
            max_body_size: 128,
            ..MessageLimits::default()
        };

        let mut buffer = message_declaring_body(4096);
        assert!(
            MessageDecoder::new(limits).read_next(&mut buffer).is_err(),
            "a body above the configured limit must be rejected"
        );

        let mut buffer = message_declaring_body(4096);
        match MessageDecoder::new(MessageLimits::UNLIMITED)
            .read_next(&mut buffer)
            .unwrap()
        {
            PollRead::NeedMore(n) => assert_eq!(n, 4096),
            otherwise => vortex_panic!("Expected NeedMore, got {:?}", otherwise),
        }
    }
}

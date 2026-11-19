// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoding serialized protobuf messages into Vortex arrays.
//!
//! The decoder reads the wire format straight into one builder per storage column, without a
//! message object per row. When it is created, it compiles the message descriptor into a tree of
//! columns that know their field numbers, wire types and oneofs. Decoding a message then walks
//! its fields once, in the order they appear, and appends each value to its column.
//!
//! It follows protobuf parse semantics:
//!
//! - Fields may appear in any order, and unknown field numbers are skipped, with groups.
//! - The last value of a singular scalar field wins, and so does the last member of a oneof.
//! - A singular message field that appears again merges into the message, as if the two encodings
//!   were concatenated.
//! - A repeated scalar field accepts packed and unpacked values, mixed.
//! - A map entry without a key or value takes the default, and a later entry replaces an earlier
//!   one with the same key.
//! - A value whose wire type does not match its field is skipped like an unknown field.
//! - A number that a closed (proto2) enum does not define is skipped like an unknown field. A map
//!   entry with such a value is dropped.
//!
//! If the mapping preserves unknown fields, each message collects the bytes of the fields that it
//! skips, in the order read. A number of a packed run that a closed enum does not define becomes
//! a varint field of its own, as in protobuf parsers.
//!
//! A message beyond the recursion depth is not parsed: its occurrences are appended to its bytes,
//! which merges them as protobuf does. A well-known type in a dedicated form collects its bytes in
//! the same way, and converts them once the whole top-level message has decoded, so that the
//! occurrences have merged.
//!
//! A column that a message does not set falls behind and catches up with one call that appends
//! all absent rows, when a later message sets it or when the decoder finishes. So the cost of a
//! message is proportional to the values it holds, not to the fields of the schema.

mod builders;
mod message;
mod well_known;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::dtype::extension::ExtDType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use self::message::MessageColumn;
use crate::Protobuf;

/// Decodes serialized protobuf messages into an array of the extension dtype `dtype`.
///
/// This is a shorthand for a [`ProtobufDecoder`] that appends every message.
///
/// # Errors
///
/// Returns an error if a message is not a valid encoding of the dtype's message type.
pub fn decode_messages<'a>(
    dtype: &ExtDType<Protobuf>,
    messages: impl IntoIterator<Item = &'a [u8]>,
) -> VortexResult<ArrayRef> {
    let messages = messages.into_iter();
    let mut decoder = ProtobufDecoder::with_capacity(dtype, messages.size_hint().0);
    for message in messages {
        decoder.append_message(message)?;
    }
    decoder.finish()
}

/// Builds an array of a protobuf extension dtype from serialized messages.
///
/// Each appended message is decoded from the protobuf wire format into one builder per storage
/// column. A field that is absent from a message becomes null if it has explicit presence, and
/// takes its protobuf default value otherwise. The [module documentation](self) lists the parse
/// semantics.
pub struct ProtobufDecoder {
    /// The extension dtype of the result.
    dtype: ExtDType<Protobuf>,
    /// The columns of the root message.
    root: MessageColumn,
}

impl ProtobufDecoder {
    /// Creates a decoder for arrays of the extension dtype `dtype`.
    pub fn new(dtype: &ExtDType<Protobuf>) -> Self {
        Self::with_capacity(dtype, 0)
    }

    /// Creates a decoder with room for `capacity` messages.
    pub fn with_capacity(dtype: &ExtDType<Protobuf>, capacity: usize) -> Self {
        Self {
            dtype: dtype.clone(),
            root: MessageColumn::new(
                dtype.metadata().plan(),
                dtype.storage_dtype().nullability(),
                capacity,
            ),
        }
    }

    /// Appends one serialized message.
    ///
    /// # Errors
    ///
    /// Returns an error if `message` is not a valid encoding of the dtype's message type: if it is
    /// truncated or malformed, or a string field holds invalid UTF-8. It also fails if a
    /// well-known type has no stored form under the mapping options, such as a timestamp outside
    /// the nanosecond range. The decoder is unchanged in that case.
    pub fn append_message(&mut self, message: &[u8]) -> VortexResult<()> {
        let row = self.root.len();
        self.root.enter(row);
        let mut input = message;
        let decoded = self
            .root
            .merge(row, &mut input, None)
            .and_then(|()| self.root.convert(row));
        if let Err(err) = decoded {
            self.root.truncate(row);
            return Err(err.with_context(format!(
                "Invalid serialized {} message",
                self.dtype.metadata().message_name()
            )));
        }
        Ok(())
    }

    /// Appends a null message.
    ///
    /// # Errors
    ///
    /// Returns an error if the dtype is not nullable.
    pub fn append_null(&mut self) -> VortexResult<()> {
        if !self.dtype.storage_dtype().is_nullable() {
            vortex_bail!("Cannot append a null message to a non-nullable protobuf array");
        }
        self.root.fill(self.root.len() + 1);
        Ok(())
    }

    /// The number of messages appended so far.
    pub fn len(&self) -> usize {
        self.root.len()
    }

    /// Whether no messages were appended.
    pub fn is_empty(&self) -> bool {
        self.root.len() == 0
    }

    /// Finishes the array of all appended messages.
    ///
    /// # Errors
    ///
    /// Returns an error if the decoded storage does not match the dtype, which indicates a bug.
    pub fn finish(self) -> VortexResult<ArrayRef> {
        let len = self.root.len();
        let storage = self.root.finish(len)?;
        Ok(ExtensionArray::try_new(self.dtype.erased(), storage)?.into_array())
    }
}

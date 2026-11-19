// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native scalar values of the protobuf extension dtype.

use std::fmt;

use prost_reflect::DynamicMessage;
use prost_reflect::MessageDescriptor;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::ProtobufMetadata;
use crate::encode::encode_value;

/// A non-null protobuf message scalar, viewed through its storage value.
///
/// [`to_message`](Self::to_message) converts the storage value back to a protobuf message, and
/// [`encode_to_vec`](Self::encode_to_vec) to its wire format.
#[derive(Clone, Copy, Debug)]
pub struct ProtobufValue<'a> {
    /// The metadata of the dtype, with the message type and the storage mapping.
    metadata: &'a ProtobufMetadata,
    /// The storage struct value, with one entry per storage field.
    storage: &'a ScalarValue,
}

impl<'a> ProtobufValue<'a> {
    /// Views `storage` as a message of the dtype with `metadata`.
    pub(crate) fn try_new(
        metadata: &'a ProtobufMetadata,
        storage: &'a ScalarValue,
    ) -> VortexResult<Self> {
        let ScalarValue::Tuple(fields) = storage else {
            vortex_bail!("Protobuf storage scalar must be a struct, got {storage}");
        };
        vortex_ensure_eq!(fields.len(), metadata.plan().storage_fields());
        Ok(Self { metadata, storage })
    }

    /// The message type, from the pool of
    /// [`ProtobufMetadata::message_descriptor`](crate::ProtobufMetadata::message_descriptor).
    pub fn message_descriptor(&self) -> &'a MessageDescriptor {
        self.metadata.message_descriptor()
    }

    /// The storage struct value, with one entry per storage field.
    pub fn storage_value(&self) -> &'a ScalarValue {
        self.storage
    }

    /// Converts the value to a protobuf message.
    ///
    /// This parses the [encoding](Self::encode_to_vec) with `prost-reflect`, so fields with
    /// implicit presence that hold their default value are unset, as in a decoded message, and
    /// stored unknown fields become unknown fields of the message. The message has the type
    /// [`message_descriptor`](Self::message_descriptor), which belongs to the pool that the dtype
    /// metadata builds from its descriptor set. So it equals the message that decoding the wire
    /// format with that descriptor gives, but never a message of the caller's own descriptor,
    /// because descriptors of different pools are never equal. Compare with such a message by the
    /// encoded bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage value does not follow the storage mapping, or if
    /// `prost-reflect` rejects the encoding. It rejects stored unknown fields that have the
    /// number of a known field but another wire type, which protobuf parsers keep as unknown.
    pub fn to_message(&self) -> VortexResult<DynamicMessage> {
        let encoded = self.encode_to_vec()?;
        DynamicMessage::decode(self.message_descriptor().clone(), encoded.as_slice())
            .map_err(|err| vortex_err!("prost-reflect cannot parse the message: {err}"))
    }

    /// Encodes the value in the protobuf wire format, as
    /// [`encode_messages`](crate::encode_messages) encodes it.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage value does not follow the storage mapping.
    pub fn encode_to_vec(&self) -> VortexResult<Vec<u8>> {
        encode_value(self.metadata.plan(), self.storage)
    }
}

impl fmt::Display for ProtobufValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.storage, f)
    }
}

/// Returns the non-null elements of a stored list.
pub(crate) fn list_elements(
    storage: &ScalarValue,
) -> VortexResult<impl Iterator<Item = VortexResult<&ScalarValue>>> {
    let ScalarValue::Tuple(elements) = storage else {
        vortex_bail!("Repeated protobuf fields must be stored as lists, got {storage}");
    };
    Ok(elements.iter().map(|element| {
        element
            .as_ref()
            .ok_or_else(|| vortex_err!("Repeated protobuf fields cannot hold null elements"))
    }))
}

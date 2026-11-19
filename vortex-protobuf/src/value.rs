// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native scalar values of the protobuf extension dtype.

use std::fmt;

use prost::Message;
use prost::bytes::Bytes;
use prost_reflect::DynamicMessage;
use prost_reflect::FieldDescriptor;
use prost_reflect::Kind;
use prost_reflect::MapKey;
use prost_reflect::MessageDescriptor;
use prost_reflect::Value;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::mapping::FieldShape;

/// A non-null protobuf message scalar, viewed through its storage value.
///
/// [`to_message`](Self::to_message) converts the storage value back to a protobuf message, and
/// [`encode_to_vec`](Self::encode_to_vec) to its wire format.
#[derive(Clone, Copy, Debug)]
pub struct ProtobufValue<'a> {
    /// The message type.
    descriptor: &'a MessageDescriptor,
    /// The storage struct value, with one entry per message field.
    storage: &'a ScalarValue,
}

impl<'a> ProtobufValue<'a> {
    /// Views `storage` as a message of type `descriptor`.
    pub(crate) fn try_new(
        descriptor: &'a MessageDescriptor,
        storage: &'a ScalarValue,
    ) -> VortexResult<Self> {
        let ScalarValue::Tuple(fields) = storage else {
            vortex_bail!("Protobuf storage scalar must be a struct, got {storage}");
        };
        vortex_ensure_eq!(fields.len(), descriptor.fields().len());
        Ok(Self {
            descriptor,
            storage,
        })
    }

    /// The message type, from the pool of
    /// [`ProtobufMetadata::message_descriptor`](crate::ProtobufMetadata::message_descriptor).
    pub fn message_descriptor(&self) -> &'a MessageDescriptor {
        self.descriptor
    }

    /// The storage struct value, with one entry per message field.
    pub fn storage_value(&self) -> &'a ScalarValue {
        self.storage
    }

    /// Converts the value to a protobuf message.
    ///
    /// Fields with implicit presence that hold their default value are unset, as in a decoded
    /// message. The message has the type [`message_descriptor`](Self::message_descriptor), which
    /// belongs to the pool that the dtype metadata builds from its descriptor set. So it equals
    /// the message that decoding the wire format with that descriptor gives, but never a message
    /// of the caller's own descriptor, because descriptors of different pools are never equal.
    /// Compare with such a message by the encoded bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage value does not follow the storage mapping.
    pub fn to_message(&self) -> VortexResult<DynamicMessage> {
        message_from_storage(self.descriptor, self.storage)
    }

    /// Encodes the value in the protobuf wire format.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage value does not follow the storage mapping.
    pub fn encode_to_vec(&self) -> VortexResult<Vec<u8>> {
        Ok(self.to_message()?.encode_to_vec())
    }
}

impl fmt::Display for ProtobufValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.storage, f)
    }
}

/// Converts a storage struct value to a message of type `descriptor`.
fn message_from_storage(
    descriptor: &MessageDescriptor,
    storage: &ScalarValue,
) -> VortexResult<DynamicMessage> {
    let ScalarValue::Tuple(values) = storage else {
        vortex_bail!(
            "Protobuf message {} must be stored as a struct, got {storage}",
            descriptor.full_name()
        );
    };
    vortex_ensure_eq!(values.len(), descriptor.fields().len());

    let mut message = DynamicMessage::new(descriptor.clone());
    for (field, value) in descriptor.fields().zip(values) {
        let Some(value) = value else {
            continue;
        };
        let value = field_from_storage(&field, value)?;
        // A decoded message does not hold default values of fields with implicit presence, but
        // it holds a `-0.0`, which equals the default zero.
        if field.supports_presence()
            || !value.is_default_for_field(&field)
            || is_negative_zero(&value)
        {
            message.set_field(&field, value);
        }
    }
    Ok(message)
}

/// Whether `value` is a floating point negative zero.
fn is_negative_zero(value: &Value) -> bool {
    match value {
        Value::F64(value) => value.to_bits() == (-0.0f64).to_bits(),
        Value::F32(value) => value.to_bits() == (-0.0f32).to_bits(),
        _ => false,
    }
}

/// Converts the non-null storage value of `field` to a protobuf value.
fn field_from_storage(field: &FieldDescriptor, storage: &ScalarValue) -> VortexResult<Value> {
    match FieldShape::of(field) {
        FieldShape::Single(kind) => value_from_storage(&kind, storage),
        FieldShape::List(kind) => list_elements(storage)?
            .map(|element| value_from_storage(&kind, element?))
            .collect::<VortexResult<_>>()
            .map(Value::List),
        FieldShape::Map { key, value } => list_elements(storage)?
            .map(|entry| {
                let ScalarValue::Tuple(entry) = entry? else {
                    vortex_bail!("Protobuf map entries must be stored as structs");
                };
                let [Some(key_storage), Some(value_storage)] = entry.as_slice() else {
                    vortex_bail!("Protobuf map entries must have a non-null key and value");
                };
                Ok((
                    map_key(value_from_storage(&key, key_storage)?)?,
                    value_from_storage(&value, value_storage)?,
                ))
            })
            .collect::<VortexResult<_>>()
            .map(Value::Map),
    }
}

/// Returns the non-null elements of a stored list.
fn list_elements(
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

/// Converts a protobuf value to a map key.
pub(crate) fn map_key(value: Value) -> VortexResult<MapKey> {
    Ok(match value {
        Value::Bool(key) => MapKey::Bool(key),
        Value::I32(key) => MapKey::I32(key),
        Value::I64(key) => MapKey::I64(key),
        Value::U32(key) => MapKey::U32(key),
        Value::U64(key) => MapKey::U64(key),
        Value::String(key) => MapKey::String(key),
        value => vortex_bail!("Protobuf value {value:?} cannot be a map key"),
    })
}

/// Converts the non-null storage value of a single protobuf value of `kind`.
fn value_from_storage(kind: &Kind, storage: &ScalarValue) -> VortexResult<Value> {
    Ok(match (kind, storage) {
        (Kind::Message(message), _) => Value::Message(message_from_storage(message, storage)?),
        (Kind::Bool, ScalarValue::Bool(value)) => Value::Bool(*value),
        (Kind::String, ScalarValue::Utf8(value)) => Value::String(value.as_str().to_owned()),
        (Kind::Bytes, ScalarValue::Binary(value)) => {
            Value::Bytes(Bytes::copy_from_slice(value.as_slice()))
        }
        (Kind::Double, ScalarValue::Primitive(value)) => Value::F64(value.cast()?),
        (Kind::Float, ScalarValue::Primitive(value)) => Value::F32(value.cast()?),
        (Kind::Int32 | Kind::Sint32 | Kind::Sfixed32, ScalarValue::Primitive(value)) => {
            Value::I32(value.cast()?)
        }
        (Kind::Enum(_), ScalarValue::Primitive(value)) => Value::EnumNumber(value.cast()?),
        (Kind::Int64 | Kind::Sint64 | Kind::Sfixed64, ScalarValue::Primitive(value)) => {
            Value::I64(value.cast()?)
        }
        (Kind::Uint32 | Kind::Fixed32, ScalarValue::Primitive(value)) => Value::U32(value.cast()?),
        (Kind::Uint64 | Kind::Fixed64, ScalarValue::Primitive(value)) => Value::U64(value.cast()?),
        _ => vortex_bail!("Protobuf {kind:?} values cannot be stored as {storage}"),
    })
}

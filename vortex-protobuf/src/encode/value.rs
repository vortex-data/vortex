// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encoding one message from its storage scalar value.

use prost::encoding::WireType;
use prost_reflect::FieldDescriptor;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;

use super::scalar::ScalarRef;
use super::scalar::write_key;
use super::scalar::write_len;
use super::scalar::write_wrapper;
use crate::mapping::FieldPlan;
use crate::mapping::FieldShape;
use crate::mapping::MessagePlan;
use crate::mapping::ValuePlan;
use crate::value::list_elements;
use crate::well_known::WellKnown;
use crate::well_known::encode_json;
use crate::well_known::encode_timestamp;
use crate::wire::ScalarKind;

/// Encodes the storage struct value of a message that `plan` describes in the wire format.
///
/// The fields are written in field number order, followed by the unknown fields, as
/// [`encode_messages`](crate::encode_messages) writes them.
///
/// # Errors
///
/// Returns an error if the storage value does not follow the storage mapping.
pub(crate) fn encode_value(plan: &MessagePlan, storage: &ScalarValue) -> VortexResult<Vec<u8>> {
    let mut buf = Vec::new();
    write_message(plan, storage, &mut buf)?;
    Ok(buf)
}

/// Writes the fields of the stored message `storage` that `plan` describes.
fn write_message(plan: &MessagePlan, storage: &ScalarValue, buf: &mut Vec<u8>) -> VortexResult<()> {
    let ScalarValue::Tuple(values) = storage else {
        vortex_bail!(
            "Protobuf message {} must be stored as a struct, got {storage}",
            plan.descriptor.full_name()
        );
    };
    vortex_ensure_eq!(values.len(), plan.storage_fields());
    let mut fields = plan.fields.iter().zip(values).collect::<Vec<_>>();
    fields.sort_unstable_by_key(|(field, _)| field.descriptor.number());
    for (field, value) in fields {
        if let Some(value) = value {
            write_field(field, value, buf)?;
        }
    }
    if plan.unknown_fields {
        match values.last() {
            Some(Some(ScalarValue::Binary(unknown))) => buf.extend_from_slice(unknown.as_slice()),
            Some(None) => {}
            Some(Some(value)) => {
                vortex_bail!("Unknown protobuf fields must be binary, got {value}")
            }
            None => vortex_bail!("Protobuf message storage has no unknown fields"),
        }
    }
    Ok(())
}

/// Writes the non-null storage value of the field that `plan` describes.
fn write_field(plan: &FieldPlan, value: &ScalarValue, buf: &mut Vec<u8>) -> VortexResult<()> {
    let field = &plan.descriptor;
    let number = field.number();
    match &plan.shape {
        FieldShape::Single(value_plan) => write_single(field, value_plan, value, buf),
        FieldShape::List(ValuePlan::Scalar { kind, .. }) => {
            let elements = list_elements(value)?
                .map(|element| scalar_ref(*kind, element?))
                .collect::<VortexResult<Vec<_>>>()?;
            if elements.is_empty() {
                return Ok(());
            }
            if field.is_packed() {
                write_key(number, WireType::LengthDelimited, buf);
                write_len(
                    elements
                        .iter()
                        .map(|element| element.encoded_len(*kind))
                        .sum(),
                    buf,
                );
                for element in elements {
                    element.write(*kind, buf);
                }
            } else {
                for element in elements {
                    write_key(number, kind.wire_type(), buf);
                    element.write(*kind, buf);
                }
            }
            Ok(())
        }
        FieldShape::List(value_plan) => {
            for element in list_elements(value)? {
                write_message_field(number, field.is_group(), value_plan, element?, buf)?;
            }
            Ok(())
        }
        FieldShape::Map {
            entry,
            key,
            value: value_plan,
        } => {
            let key_plan = ValuePlan::Scalar {
                kind: *key,
                enum_type: None,
            };
            let (key_field, value_field) =
                (entry.map_entry_key_field(), entry.map_entry_value_field());
            let mut entry_buf = Vec::new();
            for entry in list_elements(value)? {
                let ScalarValue::Tuple(entry) = entry? else {
                    vortex_bail!("Protobuf map entries must be stored as structs");
                };
                let [Some(key), Some(value)] = entry.as_slice() else {
                    vortex_bail!("Protobuf map entries must have a non-null key and value");
                };
                entry_buf.clear();
                write_single(&key_field, &key_plan, key, &mut entry_buf)?;
                write_single(&value_field, value_plan, value, &mut entry_buf)?;
                write_key(number, WireType::LengthDelimited, buf);
                write_len(entry_buf.len(), buf);
                buf.extend_from_slice(&entry_buf);
            }
            Ok(())
        }
    }
}

/// Writes the non-null storage value of `field`, which holds at most one value that `plan`
/// describes.
fn write_single(
    field: &FieldDescriptor,
    plan: &ValuePlan,
    value: &ScalarValue,
    buf: &mut Vec<u8>,
) -> VortexResult<()> {
    let ValuePlan::Scalar { kind, .. } = plan else {
        return write_message_field(field.number(), field.is_group(), plan, value, buf);
    };
    let value = scalar_ref(*kind, value)?;
    if field.supports_presence() || !value.is_default() {
        write_key(field.number(), kind.wire_type(), buf);
        value.write(*kind, buf);
    }
    Ok(())
}

/// Writes the stored message `value` that `plan` describes as an occurrence of the field
/// `number`, which is a group if `group` is set.
fn write_message_field(
    number: u32,
    group: bool,
    plan: &ValuePlan,
    value: &ScalarValue,
    buf: &mut Vec<u8>,
) -> VortexResult<()> {
    let nested = message_encoding(plan, value)?;
    if group {
        write_key(number, WireType::StartGroup, buf);
        buf.extend_from_slice(&nested);
        write_key(number, WireType::EndGroup, buf);
    } else {
        write_key(number, WireType::LengthDelimited, buf);
        write_len(nested.len(), buf);
        buf.extend_from_slice(&nested);
    }
    Ok(())
}

/// Returns the encoded fields of the stored message `value` that `plan` describes.
fn message_encoding(plan: &ValuePlan, value: &ScalarValue) -> VortexResult<Vec<u8>> {
    let mut buf = Vec::new();
    match (plan, value) {
        (ValuePlan::Message(message), value) => write_message(message, value, &mut buf)?,
        (ValuePlan::Encoded, ScalarValue::Binary(encoded)) => {
            buf.extend_from_slice(encoded.as_slice());
        }
        (ValuePlan::WellKnown(WellKnown::Wrapper(kind)), value) => {
            write_wrapper(*kind, scalar_ref(*kind, value)?, &mut buf);
        }
        (ValuePlan::WellKnown(WellKnown::Timestamp), ScalarValue::Primitive(nanos)) => {
            encode_timestamp(nanos.cast()?, &mut buf);
        }
        (ValuePlan::WellKnown(WellKnown::Json(json_type)), ScalarValue::Utf8(text)) => {
            encode_json(*json_type, text.as_bytes(), &mut buf)?;
        }
        (ValuePlan::Encoded, value) => vortex_bail!(
            "Protobuf messages beyond the recursion depth must be stored as binary, got {value}"
        ),
        (ValuePlan::WellKnown(well_known), value) => {
            vortex_bail!("Protobuf {well_known:?} values cannot be stored as {value}")
        }
        (ValuePlan::Scalar { kind, .. }, _) => {
            vortex_bail!("Protobuf {kind:?} values are not messages")
        }
    }
    Ok(buf)
}

/// Reads the non-null stored value of a protobuf value of `kind`.
fn scalar_ref(kind: ScalarKind, value: &ScalarValue) -> VortexResult<ScalarRef<'_>> {
    Ok(match (kind, value) {
        (ScalarKind::Double, ScalarValue::Primitive(value)) => ScalarRef::F64(value.cast()?),
        (ScalarKind::Float, ScalarValue::Primitive(value)) => ScalarRef::F32(value.cast()?),
        (
            ScalarKind::Int32 | ScalarKind::Sint32 | ScalarKind::Sfixed32 | ScalarKind::Enum,
            ScalarValue::Primitive(value),
        ) => ScalarRef::I32(value.cast()?),
        (
            ScalarKind::Int64 | ScalarKind::Sint64 | ScalarKind::Sfixed64,
            ScalarValue::Primitive(value),
        ) => ScalarRef::I64(value.cast()?),
        (ScalarKind::Uint32 | ScalarKind::Fixed32, ScalarValue::Primitive(value)) => {
            ScalarRef::U32(value.cast()?)
        }
        (ScalarKind::Uint64 | ScalarKind::Fixed64, ScalarValue::Primitive(value)) => {
            ScalarRef::U64(value.cast()?)
        }
        (ScalarKind::Bool, ScalarValue::Bool(value)) => ScalarRef::Bool(*value),
        (ScalarKind::String, ScalarValue::Utf8(value)) => ScalarRef::Bytes(value.as_bytes()),
        (ScalarKind::Bytes, ScalarValue::Binary(value)) => ScalarRef::Bytes(value.as_slice()),
        _ => vortex_bail!("Protobuf {kind:?} values cannot be stored as {value}"),
    })
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Version 1 of the mapping from protobuf message descriptors to Vortex storage dtypes.

use std::sync::Arc;

use prost_reflect::FieldDescriptor;
use prost_reflect::Kind;
use prost_reflect::MessageDescriptor;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

/// How a protobuf field stores its values.
pub(crate) enum FieldShape {
    /// A singular field.
    Single(Kind),
    /// A `repeated` field that is not a map.
    List(Kind),
    /// A `map<K, V>` field, stored as a list of `{key, value}` entries.
    Map {
        /// The kind of the map keys.
        key: Kind,
        /// The kind of the map values.
        value: Kind,
    },
}

impl FieldShape {
    /// Classifies `field` by cardinality.
    pub(crate) fn of(field: &FieldDescriptor) -> Self {
        if field.is_map() {
            let entry = field
                .kind()
                .as_message()
                .cloned()
                .vortex_expect("a map field has a map entry message");
            Self::Map {
                key: entry.map_entry_key_field().kind(),
                value: entry.map_entry_value_field().kind(),
            }
        } else if field.is_list() {
            Self::List(field.kind())
        } else {
            Self::Single(field.kind())
        }
    }
}

/// The names of the fields of a stored map entry.
pub(crate) const MAP_ENTRY_FIELDS: [&str; 2] = ["key", "value"];

/// Returns the storage dtype of `message` with the given top-level nullability.
///
/// The message must not be recursive, or this does not terminate.
pub(crate) fn message_dtype(message: &MessageDescriptor, nullability: Nullability) -> DType {
    let fields = message
        .fields()
        .map(|field| (FieldName::from(field.name()), field_dtype(&field)))
        .collect::<StructFields>();
    DType::Struct(fields, nullability)
}

/// Returns the storage dtype of one message field.
pub(crate) fn field_dtype(field: &FieldDescriptor) -> DType {
    match FieldShape::of(field) {
        FieldShape::Single(kind) => kind_dtype(&kind, field.supports_presence().into()),
        FieldShape::List(kind) => DType::List(
            Arc::new(kind_dtype(&kind, Nullability::NonNullable)),
            Nullability::NonNullable,
        ),
        FieldShape::Map { key, value } => {
            let entry = MAP_ENTRY_FIELDS
                .into_iter()
                .zip([
                    kind_dtype(&key, Nullability::NonNullable),
                    kind_dtype(&value, Nullability::NonNullable),
                ])
                .collect::<StructFields>();
            DType::List(
                Arc::new(DType::Struct(entry, Nullability::NonNullable)),
                Nullability::NonNullable,
            )
        }
    }
}

/// Returns the storage dtype of a single protobuf value of `kind`.
pub(crate) fn kind_dtype(kind: &Kind, nullability: Nullability) -> DType {
    match kind {
        Kind::Message(message) => message_dtype(message, nullability),
        Kind::Bool => DType::Bool(nullability),
        Kind::String => DType::Utf8(nullability),
        Kind::Bytes => DType::Binary(nullability),
        _ => DType::Primitive(kind_ptype(kind), nullability),
    }
}

/// Returns the primitive type of a numeric or enum `kind`.
///
/// # Panics
///
/// Panics if `kind` is a message, `bool`, `string` or `bytes`.
fn kind_ptype(kind: &Kind) -> PType {
    match kind {
        Kind::Double => PType::F64,
        Kind::Float => PType::F32,
        Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 | Kind::Enum(_) => PType::I32,
        Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 => PType::I64,
        Kind::Uint32 | Kind::Fixed32 => PType::U32,
        Kind::Uint64 | Kind::Fixed64 => PType::U64,
        Kind::Bool | Kind::String | Kind::Bytes | Kind::Message(_) => {
            vortex_panic!("protobuf {kind:?} values are not primitive")
        }
    }
}

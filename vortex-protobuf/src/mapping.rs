// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The mapping from protobuf message descriptors to Vortex storage dtypes.
//!
//! A [`MessagePlan`] records how each field of a message type and of the messages that it holds
//! is stored, for one set of [`MappingOptions`]. The storage dtype, the decoder and the encoder
//! all follow the plan, so they agree on the mapping. The plan is a tree: a message type that
//! occurs at several places of the schema, or inside itself, gets a plan at each place.

use std::sync::Arc;

use prost_reflect::EnumDescriptor;
use prost_reflect::FieldDescriptor;
use prost_reflect::Kind;
use prost_reflect::MessageDescriptor;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::MappingOptions;
use crate::well_known::WellKnown;
use crate::wire::ScalarKind;

/// The names of the fields of a stored map entry.
pub(crate) const MAP_ENTRY_FIELDS: [&str; 2] = ["key", "value"];

/// The most fields that the storage of a message can have, counting the fields of nested
/// messages.
///
/// A message type that occurs at several places of a schema is stored at each, and a recursion
/// depth multiplies the fields of a recursive message. So a small descriptor set can expand to
/// more fields than memory holds, with or without recursion. The limit applies to every mapping
/// version, because stored metadata comes from files that need not be trusted, and makes such a
/// schema fail to load quickly.
pub(crate) const MAX_STORAGE_FIELDS: usize = 100_000;

/// The name of the storage field that holds the unknown fields of a message.
///
/// Protobuf names cannot contain `$`, so no message field has this name.
pub const UNKNOWN_FIELDS: &str = "$unknown";

/// How the messages of one type, at one place of the schema, are stored: as a struct with one
/// field per message field in declaration order, followed by [`UNKNOWN_FIELDS`] if the options
/// preserve unknown fields.
pub(crate) struct MessagePlan {
    /// The message type.
    pub(crate) descriptor: MessageDescriptor,
    /// The fields, in declaration order.
    pub(crate) fields: Vec<FieldPlan>,
    /// Whether the unknown fields are stored.
    pub(crate) unknown_fields: bool,
}

/// How one message field is stored.
pub(crate) struct FieldPlan {
    /// The field.
    pub(crate) descriptor: FieldDescriptor,
    /// The cardinality of the field and how its values are stored.
    pub(crate) shape: FieldShape,
}

/// The cardinality of a field and how its values are stored.
pub(crate) enum FieldShape {
    /// A singular field, nullable if the field has explicit presence.
    Single(ValuePlan),
    /// A `repeated` field that is not a map: a non-nullable list of non-nullable values.
    List(ValuePlan),
    /// A `map<K, V>` field: a non-nullable list of non-nullable `{key, value}` structs.
    Map {
        /// The map entry message.
        entry: MessageDescriptor,
        /// The kind of the keys.
        key: ScalarKind,
        /// How the values are stored.
        value: ValuePlan,
    },
}

/// How single values of a field are stored.
pub(crate) enum ValuePlan {
    /// A scalar value.
    Scalar {
        /// The kind of the value.
        kind: ScalarKind,
        /// The enum type, if the value is an enum number.
        enum_type: Option<EnumDescriptor>,
    },
    /// A message, as a struct.
    Message(Box<MessagePlan>),
    /// A message beyond the recursion depth, as the binary encoding of its fields.
    Encoded,
    /// A well-known message type in a dedicated form.
    WellKnown(WellKnown),
}

impl MessagePlan {
    /// Plans the storage of messages of type `descriptor` under `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if the message is recursive and the options do not set a recursion depth,
    /// or if the storage would have more than [`MAX_STORAGE_FIELDS`] fields.
    pub(crate) fn try_new(
        descriptor: &MessageDescriptor,
        options: &MappingOptions,
    ) -> VortexResult<Self> {
        PlanBuilder {
            options,
            path: Vec::new(),
            fields: 0,
        }
        .message(descriptor)
    }

    /// The storage dtype of the messages with the given top-level nullability.
    pub(crate) fn dtype(&self, nullability: Nullability) -> DType {
        let mut fields = self
            .fields
            .iter()
            .map(|field| (FieldName::from(field.descriptor.name()), field.dtype()))
            .collect::<Vec<_>>();
        if self.unknown_fields {
            fields.push((
                FieldName::from(UNKNOWN_FIELDS),
                DType::Binary(Nullability::NonNullable),
            ));
        }
        DType::Struct(fields.into_iter().collect::<StructFields>(), nullability)
    }

    /// The number of fields of the storage struct.
    pub(crate) fn storage_fields(&self) -> usize {
        self.fields.len() + usize::from(self.unknown_fields)
    }
}

impl FieldPlan {
    /// The storage dtype of the field.
    pub(crate) fn dtype(&self) -> DType {
        match &self.shape {
            FieldShape::Single(value) => value.dtype(self.descriptor.supports_presence().into()),
            FieldShape::List(value) => DType::List(
                Arc::new(value.dtype(Nullability::NonNullable)),
                Nullability::NonNullable,
            ),
            FieldShape::Map { key, value, .. } => {
                let entry = MAP_ENTRY_FIELDS
                    .into_iter()
                    .zip([
                        scalar_dtype(*key, Nullability::NonNullable),
                        value.dtype(Nullability::NonNullable),
                    ])
                    .collect::<StructFields>();
                DType::List(
                    Arc::new(DType::Struct(entry, Nullability::NonNullable)),
                    Nullability::NonNullable,
                )
            }
        }
    }
}

impl ValuePlan {
    /// The storage dtype of a value with the given nullability.
    pub(crate) fn dtype(&self, nullability: Nullability) -> DType {
        match self {
            Self::Scalar { kind, .. } => scalar_dtype(*kind, nullability),
            Self::Message(message) => message.dtype(nullability),
            Self::Encoded => DType::Binary(nullability),
            Self::WellKnown(well_known) => well_known.dtype(nullability),
        }
    }
}

/// Returns the storage dtype of a scalar value of `kind`.
pub(crate) fn scalar_dtype(kind: ScalarKind, nullability: Nullability) -> DType {
    let ptype = match kind {
        ScalarKind::Double => PType::F64,
        ScalarKind::Float => PType::F32,
        ScalarKind::Int32 | ScalarKind::Sint32 | ScalarKind::Sfixed32 | ScalarKind::Enum => {
            PType::I32
        }
        ScalarKind::Int64 | ScalarKind::Sint64 | ScalarKind::Sfixed64 => PType::I64,
        ScalarKind::Uint32 | ScalarKind::Fixed32 => PType::U32,
        ScalarKind::Uint64 | ScalarKind::Fixed64 => PType::U64,
        ScalarKind::Bool => return DType::Bool(nullability),
        ScalarKind::String => return DType::Utf8(nullability),
        ScalarKind::Bytes => return DType::Binary(nullability),
    };
    DType::Primitive(ptype, nullability)
}

/// Builds the plan of a message type, depth first.
struct PlanBuilder<'a> {
    /// The mapping options.
    options: &'a MappingOptions,
    /// The messages from the root down to the one being planned, with map entry messages.
    path: Vec<MessageDescriptor>,
    /// The number of fields planned so far.
    fields: usize,
}

impl PlanBuilder<'_> {
    /// Plans the storage of `descriptor` at the end of the path.
    fn message(&mut self, descriptor: &MessageDescriptor) -> VortexResult<MessagePlan> {
        self.path.push(descriptor.clone());
        let fields = descriptor
            .fields()
            .map(|field| self.field(&field))
            .collect::<VortexResult<Vec<_>>>()?;
        self.path.pop();
        Ok(MessagePlan {
            descriptor: descriptor.clone(),
            fields,
            unknown_fields: self.options.preserve_unknown_fields,
        })
    }

    /// Plans the storage of `field`.
    fn field(&mut self, field: &FieldDescriptor) -> VortexResult<FieldPlan> {
        self.fields += 1;
        if self.fields > MAX_STORAGE_FIELDS {
            vortex_bail!(
                "The storage of protobuf message {} needs more than {MAX_STORAGE_FIELDS} fields",
                self.path[0].full_name()
            );
        }
        let shape = if field.is_map() {
            let entry = field
                .kind()
                .as_message()
                .cloned()
                .vortex_expect("a map field has a map entry message");
            let key = ScalarKind::of(&entry.map_entry_key_field().kind())
                .vortex_expect("protobuf map keys are scalars");
            self.path.push(entry.clone());
            let value = self.value(&entry.map_entry_value_field());
            self.path.pop();
            FieldShape::Map {
                entry,
                key,
                value: value?,
            }
        } else if field.is_list() {
            FieldShape::List(self.value(field)?)
        } else {
            FieldShape::Single(self.value(field)?)
        };
        Ok(FieldPlan {
            descriptor: field.clone(),
            shape,
        })
    }

    /// Plans the storage of the single values of `field`.
    fn value(&mut self, field: &FieldDescriptor) -> VortexResult<ValuePlan> {
        let message = match field.kind() {
            Kind::Message(message) => message,
            kind => {
                return Ok(ValuePlan::Scalar {
                    kind: ScalarKind::of(&kind)
                        .vortex_expect("a protobuf kind other than a message"),
                    enum_type: kind.as_enum().cloned(),
                });
            }
        };
        // A group defines its own message type, which is never a well-known type.
        if !field.is_group()
            && let Some(well_known) = WellKnown::of(&message, self.options)
        {
            return Ok(ValuePlan::WellKnown(well_known));
        }
        if let Some(start) = self.path.iter().position(|ancestor| *ancestor == message) {
            let Some(depth) = self.options.max_recursion_depth else {
                let cycle = self.path[start..]
                    .iter()
                    .chain([&message])
                    .map(MessageDescriptor::full_name)
                    .collect::<Vec<_>>()
                    .join(" -> ");
                vortex_bail!(
                    "Recursive protobuf messages need the max_recursion_depth mapping option: \
                     {cycle}"
                );
            };
            let occurrences = self.path[start..]
                .iter()
                .filter(|ancestor| **ancestor == message)
                .count();
            if occurrences > depth as usize {
                return Ok(ValuePlan::Encoded);
            }
        }
        Ok(ValuePlan::Message(Box::new(self.message(&message)?)))
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoding serialized protobuf messages into Vortex arrays.

use prost_reflect::DynamicMessage;
use prost_reflect::FieldDescriptor;
use prost_reflect::Kind;
use prost_reflect::MapKey;
use prost_reflect::MessageDescriptor;
use prost_reflect::Value;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::StructArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::BoolBuilder;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::memory::BufferAllocatorRef;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::Protobuf;
use crate::mapping::FieldShape;
use crate::mapping::MAP_ENTRY_FIELDS;

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
/// Each appended message is decoded and its fields are appended to one builder per storage
/// column. A field that is absent from a message becomes null if it has explicit presence, and
/// takes its protobuf default value otherwise.
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
                dtype.metadata().message_descriptor(),
                dtype.storage_dtype().nullability(),
                capacity,
            ),
        }
    }

    /// Appends one serialized message.
    ///
    /// # Errors
    ///
    /// Returns an error if `message` is not a valid encoding of the dtype's message type. The
    /// decoder is unchanged in that case.
    pub fn append_message(&mut self, message: &[u8]) -> VortexResult<()> {
        let descriptor = self.dtype.metadata().message_descriptor();
        let message = DynamicMessage::decode(descriptor.clone(), message).map_err(|err| {
            VortexError::from(err).with_context(format!(
                "Invalid serialized {} message",
                descriptor.full_name()
            ))
        })?;
        self.root.append_message(&message)
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
        self.root.append_defaults(1);
        Ok(())
    }

    /// The number of messages appended so far.
    pub fn len(&self) -> usize {
        self.root.len
    }

    /// Whether no messages were appended.
    pub fn is_empty(&self) -> bool {
        self.root.len == 0
    }

    /// Finishes the array of all appended messages.
    ///
    /// # Errors
    ///
    /// Returns an error if the decoded storage does not match the dtype, which indicates a bug.
    pub fn finish(self) -> VortexResult<ArrayRef> {
        let storage = self.root.finish()?;
        Ok(ExtensionArray::try_new(self.dtype.erased(), storage)?.into_array())
    }
}

/// A builder for the storage column of one protobuf value position.
enum Column {
    /// A `double` column.
    F64(PrimitiveBuilder<f64>),
    /// A `float` column.
    F32(PrimitiveBuilder<f32>),
    /// An `int32`, `sint32`, `sfixed32` or enum column.
    I32(PrimitiveBuilder<i32>),
    /// An `int64`, `sint64` or `sfixed64` column.
    I64(PrimitiveBuilder<i64>),
    /// A `uint32` or `fixed32` column.
    U32(PrimitiveBuilder<u32>),
    /// A `uint64` or `fixed64` column.
    U64(PrimitiveBuilder<u64>),
    /// A `bool` column.
    Bool(BoolBuilder),
    /// A `string` or `bytes` column.
    Bytes(VarBinViewBuilder),
    /// A message column.
    Message(MessageColumn),
    /// A repeated or map field column.
    List(ListColumn),
}

impl Column {
    /// Creates the column of `field`.
    fn for_field(field: &FieldDescriptor, capacity: usize) -> Self {
        let elements = match FieldShape::of(field) {
            FieldShape::Single(kind) => {
                return Self::for_kind(&kind, field.supports_presence().into(), capacity);
            }
            FieldShape::List(kind) => {
                ListElements::Values(Box::new(Self::for_kind(&kind, NON_NULLABLE, capacity)))
            }
            FieldShape::Map { key, value } => ListElements::Entries {
                keys: Box::new(Self::for_kind(&key, NON_NULLABLE, capacity)),
                values: Box::new(Self::for_kind(&value, NON_NULLABLE, capacity)),
            },
        };
        let mut offsets = BufferMut::with_capacity(capacity + 1);
        offsets.push(0);
        Self::List(ListColumn {
            elements,
            offsets,
            elements_len: 0,
        })
    }

    /// Creates a column of single values of `kind`.
    fn for_kind(kind: &Kind, nullability: Nullability, capacity: usize) -> Self {
        let allocator = BufferAllocatorRef::static_ref();
        match kind {
            Kind::Double => Self::F64(PrimitiveBuilder::with_capacity_in(
                nullability,
                capacity,
                allocator,
            )),
            Kind::Float => Self::F32(PrimitiveBuilder::with_capacity_in(
                nullability,
                capacity,
                allocator,
            )),
            Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 | Kind::Enum(_) => Self::I32(
                PrimitiveBuilder::with_capacity_in(nullability, capacity, allocator),
            ),
            Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 => Self::I64(
                PrimitiveBuilder::with_capacity_in(nullability, capacity, allocator),
            ),
            Kind::Uint32 | Kind::Fixed32 => Self::U32(PrimitiveBuilder::with_capacity_in(
                nullability,
                capacity,
                allocator,
            )),
            Kind::Uint64 | Kind::Fixed64 => Self::U64(PrimitiveBuilder::with_capacity_in(
                nullability,
                capacity,
                allocator,
            )),
            Kind::Bool => Self::Bool(BoolBuilder::with_capacity_in(
                nullability,
                capacity,
                allocator,
            )),
            Kind::String => Self::Bytes(VarBinViewBuilder::with_capacity_in(
                DType::Utf8(nullability),
                capacity,
                allocator.clone(),
            )),
            Kind::Bytes => Self::Bytes(VarBinViewBuilder::with_capacity_in(
                DType::Binary(nullability),
                capacity,
                allocator.clone(),
            )),
            Kind::Message(message) => {
                Self::Message(MessageColumn::new(message, nullability, capacity))
            }
        }
    }

    /// Appends a present value.
    fn append_value(&mut self, value: &Value) -> VortexResult<()> {
        match (self, value) {
            (Self::F64(builder), Value::F64(value)) => builder.append_value(*value),
            (Self::F32(builder), Value::F32(value)) => builder.append_value(*value),
            (Self::I32(builder), Value::I32(value) | Value::EnumNumber(value)) => {
                builder.append_value(*value)
            }
            (Self::I64(builder), Value::I64(value)) => builder.append_value(*value),
            (Self::U32(builder), Value::U32(value)) => builder.append_value(*value),
            (Self::U64(builder), Value::U64(value)) => builder.append_value(*value),
            (Self::Bool(builder), Value::Bool(value)) => builder.append_value(*value),
            (Self::Bytes(builder), Value::String(value)) => builder.append_value(value),
            (Self::Bytes(builder), Value::Bytes(value)) => builder.append_value(value),
            (Self::Message(column), Value::Message(message)) => column.append_message(message)?,
            (Self::List(column), Value::List(values)) => column.append_list(values)?,
            (Self::List(column), Value::Map(entries)) => column.append_map(entries.iter())?,
            (_, value) => vortex_bail!("Unexpected protobuf value {value:?}"),
        }
        Ok(())
    }

    /// Appends a present map key.
    fn append_map_key(&mut self, key: &MapKey) -> VortexResult<()> {
        match (self, key) {
            (Self::I32(builder), MapKey::I32(key)) => builder.append_value(*key),
            (Self::I64(builder), MapKey::I64(key)) => builder.append_value(*key),
            (Self::U32(builder), MapKey::U32(key)) => builder.append_value(*key),
            (Self::U64(builder), MapKey::U64(key)) => builder.append_value(*key),
            (Self::Bool(builder), MapKey::Bool(key)) => builder.append_value(*key),
            (Self::Bytes(builder), MapKey::String(key)) => builder.append_value(key),
            (_, key) => vortex_bail!("Unexpected protobuf map key {key:?}"),
        }
        Ok(())
    }

    /// Appends `n` absent values: nulls if the column is nullable, and protobuf defaults
    /// otherwise.
    fn append_defaults(&mut self, n: usize) {
        match self {
            Self::F64(builder) => builder.append_defaults(n),
            Self::F32(builder) => builder.append_defaults(n),
            Self::I32(builder) => builder.append_defaults(n),
            Self::I64(builder) => builder.append_defaults(n),
            Self::U32(builder) => builder.append_defaults(n),
            Self::U64(builder) => builder.append_defaults(n),
            Self::Bool(builder) => builder.append_defaults(n),
            Self::Bytes(builder) => builder.append_defaults(n),
            Self::Message(column) => column.append_defaults(n),
            Self::List(column) => column.append_empty(n),
        }
    }

    /// Finishes the column.
    fn finish(self) -> VortexResult<ArrayRef> {
        Ok(match self {
            Self::F64(mut builder) => builder.finish(),
            Self::F32(mut builder) => builder.finish(),
            Self::I32(mut builder) => builder.finish(),
            Self::I64(mut builder) => builder.finish(),
            Self::U32(mut builder) => builder.finish(),
            Self::U64(mut builder) => builder.finish(),
            Self::Bool(mut builder) => builder.finish(),
            Self::Bytes(mut builder) => builder.finish(),
            Self::Message(column) => column.finish()?,
            Self::List(column) => column.finish()?,
        })
    }
}

/// Non-nullable, the nullability of list elements and map entries.
const NON_NULLABLE: Nullability = Nullability::NonNullable;

/// The struct column of a message, with one child column per message field.
struct MessageColumn {
    /// The fields of the message, in declaration order, with their columns.
    fields: Vec<(FieldDescriptor, Column)>,
    /// The struct validity, or `None` if the column is non-nullable.
    validity: Option<BitBufferMut>,
    /// The number of messages in the column.
    len: usize,
}

impl MessageColumn {
    /// Creates the column of messages of type `descriptor`.
    fn new(descriptor: &MessageDescriptor, nullability: Nullability, capacity: usize) -> Self {
        Self {
            fields: descriptor
                .fields()
                .map(|field| {
                    let column = Column::for_field(&field, capacity);
                    (field, column)
                })
                .collect(),
            validity: nullability
                .is_nullable()
                .then(|| BitBufferMut::with_capacity(capacity)),
            len: 0,
        }
    }

    /// Appends a present message.
    fn append_message(&mut self, message: &DynamicMessage) -> VortexResult<()> {
        for (field, column) in &mut self.fields {
            // `has_field` treats a field without presence as absent when it equals its default, as
            // a `-0.0` does, so such a field always appends its value or default.
            if field.supports_presence() && !message.has_field(field) {
                column.append_defaults(1);
            } else {
                column.append_value(&message.get_field(field))?;
            }
        }
        self.append_validity(true, 1);
        Ok(())
    }

    /// Appends `n` absent messages: nulls if the column is nullable, and empty messages otherwise.
    fn append_defaults(&mut self, n: usize) {
        for (_, column) in &mut self.fields {
            column.append_defaults(n);
        }
        self.append_validity(false, n);
    }

    /// Records `n` messages as valid or null. A non-nullable column ignores `valid`.
    fn append_validity(&mut self, valid: bool, n: usize) {
        if let Some(validity) = &mut self.validity {
            validity.append_n(valid, n);
        }
        self.len += n;
    }

    /// Finishes the struct column.
    fn finish(self) -> VortexResult<ArrayRef> {
        let names = self
            .fields
            .iter()
            .map(|(field, _)| field.name())
            .collect::<FieldNames>();
        let fields = self
            .fields
            .into_iter()
            .map(|(_, column)| column.finish())
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = match self.validity {
            Some(validity) => Validity::from_bit_buffer(validity.freeze(), Nullability::Nullable),
            None => Validity::NonNullable,
        };
        Ok(StructArray::try_new(names, fields, self.len, validity)?.into_array())
    }
}

/// The list column of a repeated or map field.
struct ListColumn {
    /// The elements of all lists.
    elements: ListElements,
    /// The offsets of the lists into the elements, starting with zero.
    offsets: BufferMut<u64>,
    /// The number of elements of all lists.
    elements_len: usize,
}

/// The elements of a [`ListColumn`].
enum ListElements {
    /// The elements of a repeated field.
    Values(Box<Column>),
    /// The entries of a map field.
    Entries {
        /// The map keys.
        keys: Box<Column>,
        /// The map values.
        values: Box<Column>,
    },
}

impl ListColumn {
    /// Appends a list of present elements.
    fn append_list(&mut self, values: &[Value]) -> VortexResult<()> {
        let ListElements::Values(elements) = &mut self.elements else {
            vortex_bail!("Unexpected protobuf list for a map field");
        };
        for value in values {
            elements.append_value(value)?;
        }
        self.push_offset(values.len());
        Ok(())
    }

    /// Appends the entries of a map, ordered by key.
    fn append_map<'a>(
        &mut self,
        entries: impl IntoIterator<Item = (&'a MapKey, &'a Value)>,
    ) -> VortexResult<()> {
        let ListElements::Entries { keys, values } = &mut self.elements else {
            vortex_bail!("Unexpected protobuf map for a repeated field");
        };
        let mut sorted = entries.into_iter().collect::<Vec<_>>();
        sorted.sort_unstable_by_key(|&(key, _)| key);
        for &(key, value) in &sorted {
            keys.append_map_key(key)?;
            values.append_value(value)?;
        }
        self.push_offset(sorted.len());
        Ok(())
    }

    /// Appends `n` empty lists.
    fn append_empty(&mut self, n: usize) {
        self.offsets.push_n(self.elements_len as u64, n);
    }

    /// Ends a list that holds the last `len` elements.
    fn push_offset(&mut self, len: usize) {
        self.elements_len += len;
        self.offsets.push(self.elements_len as u64);
    }

    /// Finishes the list column.
    fn finish(self) -> VortexResult<ArrayRef> {
        let elements = match self.elements {
            ListElements::Values(elements) => elements.finish()?,
            ListElements::Entries { keys, values } => StructArray::try_new(
                FieldNames::from(MAP_ENTRY_FIELDS),
                [keys.finish()?, values.finish()?],
                self.elements_len,
                Validity::NonNullable,
            )?
            .into_array(),
        };
        let offsets = self.offsets.freeze().into_array();
        // SAFETY: the offsets start at zero and each one adds the number of elements appended for
        // its list to the previous one, so they never decrease and the last one is the length of
        // `elements`. The lists are non-nullable, as `Validity::NonNullable` states.
        Ok(
            unsafe { ListArray::new_unchecked(elements, offsets, Validity::NonNullable) }
                .into_array(),
        )
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encoding protobuf messages from the canonical columns of their storage.

use std::ops::Range;

use prost::encoding::WireType;
use prost_reflect::FieldDescriptor;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use super::scalar::ScalarRef;
use super::scalar::field_key_len;
use super::scalar::length_delimited_len;
use super::scalar::write_key;
use super::scalar::write_len;
use super::scalar::write_wrapper;
use crate::mapping::FieldPlan;
use crate::mapping::FieldShape;
use crate::mapping::MessagePlan;
use crate::mapping::ValuePlan;
use crate::well_known::WellKnown;
use crate::well_known::encode_json;
use crate::well_known::encode_timestamp;
use crate::wire::ScalarKind;

/// Encodes the messages of one message type from the struct column of their storage.
///
/// [`compute_sizes`](Self::compute_sizes) computes the encoded size of every message, its nested
/// messages first, so that [`write`](Self::write) can write the length of a nested message
/// before the message itself, in one pass.
pub(super) struct MessageEncoder {
    /// The fields, in field number order.
    fields: Vec<FieldEncoder>,
    /// The encoded unknown fields of each message, if the mapping stores them.
    unknown: Option<Encodings>,
    /// The rows that hold a message.
    validity: Mask,
    /// The encoded size of the message at each row, or zero at a null row.
    sizes: Vec<usize>,
}

impl MessageEncoder {
    /// Prepares to encode the messages that `plan` describes from their storage struct `array`.
    pub(super) fn new(
        plan: &MessagePlan,
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let array = array.execute::<StructArray>(ctx)?;
        let validity = array.validity()?.execute_mask(array.len(), ctx)?;
        let mut fields = plan
            .fields
            .iter()
            .enumerate()
            .map(|(index, field)| {
                FieldEncoder::new(field, array.unmasked_field(index).clone(), ctx)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        fields.sort_unstable_by_key(|field| field.number);
        let unknown = plan
            .unknown_fields
            .then(|| Encodings::stored(array.unmasked_field(plan.fields.len()).clone(), ctx))
            .transpose()?;
        Ok(Self {
            fields,
            unknown,
            validity,
            sizes: Vec::new(),
        })
    }

    /// The rows that hold a message.
    pub(super) fn validity(&self) -> &Mask {
        &self.validity
    }

    /// The encoded size of the message at `row`, which [`compute_sizes`](Self::compute_sizes)
    /// computed.
    pub(super) fn size(&self, row: usize) -> usize {
        self.sizes[row]
    }

    /// Computes the encoded size of the message at each row.
    pub(super) fn compute_sizes(&mut self) {
        for field in &mut self.fields {
            field.compute_nested_sizes();
        }
        self.sizes = (0..self.validity.len())
            .map(|row| {
                if self.validity.value(row) {
                    let unknown = self
                        .unknown
                        .as_ref()
                        .map_or(0, |unknown| unknown.get(row).len());
                    self.fields
                        .iter()
                        .map(|field| field.size(row))
                        .sum::<usize>()
                        + unknown
                } else {
                    0
                }
            })
            .collect();
    }

    /// Writes the fields of the message at `row`, which must be valid, in field number order,
    /// and then its unknown fields.
    pub(super) fn write(&self, row: usize, buf: &mut Vec<u8>) {
        for field in &self.fields {
            field.write(row, buf);
        }
        if let Some(unknown) = &self.unknown {
            buf.extend_from_slice(unknown.get(row));
        }
    }
}

/// Encodes one field of a [`MessageEncoder`].
struct FieldEncoder {
    /// The field number.
    number: u32,
    /// The storage values of the field.
    values: FieldValues,
}

/// The storage values of a [`FieldEncoder`].
enum FieldValues {
    /// A singular scalar field.
    Scalar {
        /// Whether the field has explicit presence, so that it is written even when it holds its
        /// default value.
        presence: bool,
        /// The values.
        column: ScalarColumn,
    },
    /// A singular message or group field.
    Message {
        /// Whether the field is a group.
        group: bool,
        /// The messages.
        messages: Messages,
    },
    /// A repeated scalar field.
    ScalarList {
        /// Whether the values are packed into one length-delimited run.
        packed: bool,
        /// The element range of each list.
        lists: Lists,
        /// The elements of all lists.
        elements: ScalarColumn,
    },
    /// A repeated message or group field.
    MessageList {
        /// Whether the field is a group.
        group: bool,
        /// The element range of each list.
        lists: Lists,
        /// The elements of all lists.
        elements: Messages,
    },
    /// A map field, whose entries are written as messages with a key and a value field.
    Map {
        /// The entry range of each map.
        lists: Lists,
        /// The key field of the entry message.
        key: Box<FieldEncoder>,
        /// The value field of the entry message.
        value: Box<FieldEncoder>,
    },
}

impl FieldEncoder {
    /// Prepares to encode the field that `plan` describes from its storage column `array`.
    fn new(plan: &FieldPlan, array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let field = &plan.descriptor;
        let values = match &plan.shape {
            FieldShape::Single(value) => return Self::single(field, value, array, ctx),
            FieldShape::List(value) => {
                let (lists, elements) = list_parts(array, ctx)?;
                match value {
                    ValuePlan::Scalar { kind, .. } => FieldValues::ScalarList {
                        packed: field.is_packed(),
                        lists,
                        elements: ScalarColumn::new(*kind, elements, ctx)?,
                    },
                    value => FieldValues::MessageList {
                        group: field.is_group(),
                        lists,
                        elements: Messages::new(value, elements, ctx)?,
                    },
                }
            }
            FieldShape::Map { entry, key, value } => {
                let (lists, entries) = list_parts(array, ctx)?;
                let entries = entries.execute::<StructArray>(ctx)?;
                let key = ValuePlan::Scalar {
                    kind: *key,
                    enum_type: None,
                };
                FieldValues::Map {
                    lists,
                    key: Box::new(Self::single(
                        &entry.map_entry_key_field(),
                        &key,
                        entries.unmasked_field(0).clone(),
                        ctx,
                    )?),
                    value: Box::new(Self::single(
                        &entry.map_entry_value_field(),
                        value,
                        entries.unmasked_field(1).clone(),
                        ctx,
                    )?),
                }
            }
        };
        Ok(Self {
            number: field.number(),
            values,
        })
    }

    /// Prepares to encode `field`, which holds at most one value per row that `value` describes,
    /// from `array`.
    fn single(
        field: &FieldDescriptor,
        value: &ValuePlan,
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let values = match value {
            ValuePlan::Scalar { kind, .. } => FieldValues::Scalar {
                presence: field.supports_presence(),
                column: ScalarColumn::new(*kind, array, ctx)?,
            },
            value => FieldValues::Message {
                group: field.is_group(),
                messages: Messages::new(value, array, ctx)?,
            },
        };
        Ok(Self {
            number: field.number(),
            values,
        })
    }

    /// Computes the sizes of the messages that the field holds.
    fn compute_nested_sizes(&mut self) {
        match &mut self.values {
            FieldValues::Message { messages, .. } => messages.compute_sizes(),
            FieldValues::MessageList { elements, .. } => elements.compute_sizes(),
            FieldValues::Map { value, .. } => value.compute_nested_sizes(),
            FieldValues::Scalar { .. } | FieldValues::ScalarList { .. } => {}
        }
    }

    /// The number of bytes that the field writes for `row`, keys included.
    fn size(&self, row: usize) -> usize {
        let key_len = field_key_len(self.number);
        match &self.values {
            FieldValues::Scalar { presence, column } => column
                .written(row, *presence)
                .map_or(0, |value| key_len + value.encoded_len(column.kind)),
            FieldValues::Message { group, messages } => {
                if messages.is_valid(row) {
                    nested_len(key_len, *group, messages.size(row))
                } else {
                    0
                }
            }
            FieldValues::ScalarList {
                packed,
                lists,
                elements,
            } => {
                let range = lists.range(row);
                if range.is_empty() {
                    0
                } else if *packed {
                    key_len + length_delimited_len(elements.payload_len(range))
                } else {
                    range.len() * key_len + elements.payload_len(range)
                }
            }
            FieldValues::MessageList {
                group,
                lists,
                elements,
            } => lists
                .range(row)
                .map(|element| nested_len(key_len, *group, elements.size(element)))
                .sum(),
            FieldValues::Map { lists, key, value } => lists
                .range(row)
                .map(|entry| key_len + length_delimited_len(key.size(entry) + value.size(entry)))
                .sum(),
        }
    }

    /// Writes the field for `row`, with its keys.
    fn write(&self, row: usize, buf: &mut Vec<u8>) {
        match &self.values {
            FieldValues::Scalar { presence, column } => {
                if let Some(value) = column.written(row, *presence) {
                    write_key(self.number, column.kind.wire_type(), buf);
                    value.write(column.kind, buf);
                }
            }
            FieldValues::Message { group, messages } => {
                if messages.is_valid(row) {
                    self.write_message(*group, messages, row, buf);
                }
            }
            FieldValues::ScalarList {
                packed,
                lists,
                elements,
            } => {
                let range = lists.range(row);
                if range.is_empty() {
                    return;
                }
                if *packed {
                    write_key(self.number, WireType::LengthDelimited, buf);
                    write_len(elements.payload_len(range.clone()), buf);
                    for element in range {
                        elements.get(element).write(elements.kind, buf);
                    }
                } else {
                    for element in range {
                        write_key(self.number, elements.kind.wire_type(), buf);
                        elements.get(element).write(elements.kind, buf);
                    }
                }
            }
            FieldValues::MessageList {
                group,
                lists,
                elements,
            } => {
                for element in lists.range(row) {
                    self.write_message(*group, elements, element, buf);
                }
            }
            FieldValues::Map { lists, key, value } => {
                for entry in lists.range(row) {
                    write_key(self.number, WireType::LengthDelimited, buf);
                    write_len(key.size(entry) + value.size(entry), buf);
                    key.write(entry, buf);
                    value.write(entry, buf);
                }
            }
        }
    }

    /// Writes the message at `row` of `messages` as an occurrence of this field.
    fn write_message(&self, group: bool, messages: &Messages, row: usize, buf: &mut Vec<u8>) {
        if group {
            write_key(self.number, WireType::StartGroup, buf);
            messages.write(row, buf);
            write_key(self.number, WireType::EndGroup, buf);
        } else {
            write_key(self.number, WireType::LengthDelimited, buf);
            write_len(messages.size(row), buf);
            messages.write(row, buf);
        }
    }
}

/// The messages of a message field.
enum Messages {
    /// Messages stored as structs.
    Columns(MessageEncoder),
    /// Messages whose encodings are stored, or were converted from a well-known type.
    Encoded(Encodings),
}

impl Messages {
    /// Prepares to encode the messages that `plan` describes from their storage column `array`.
    fn new(plan: &ValuePlan, array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Ok(match plan {
            ValuePlan::Message(message) => Self::Columns(MessageEncoder::new(message, array, ctx)?),
            ValuePlan::Encoded => Self::Encoded(Encodings::stored(array, ctx)?),
            ValuePlan::WellKnown(well_known) => {
                Self::Encoded(encode_well_known(*well_known, array, ctx)?)
            }
            ValuePlan::Scalar { kind, .. } => {
                vortex_panic!("Protobuf {kind:?} values are not messages")
            }
        })
    }

    /// Computes the sizes of the messages stored as structs.
    fn compute_sizes(&mut self) {
        if let Self::Columns(messages) = self {
            messages.compute_sizes();
        }
    }

    /// Whether `row` holds a message.
    fn is_valid(&self, row: usize) -> bool {
        match self {
            Self::Columns(messages) => messages.validity.value(row),
            Self::Encoded(encodings) => encodings.is_valid(row),
        }
    }

    /// The encoded size of the message at `row`.
    fn size(&self, row: usize) -> usize {
        match self {
            Self::Columns(messages) => messages.size(row),
            Self::Encoded(encodings) => encodings.get(row).len(),
        }
    }

    /// Writes the fields of the message at `row`, which must be valid.
    fn write(&self, row: usize, buf: &mut Vec<u8>) {
        match self {
            Self::Columns(messages) => messages.write(row, buf),
            Self::Encoded(encodings) => buf.extend_from_slice(encodings.get(row)),
        }
    }
}

/// Byte strings with the encoded fields of messages, one per row.
enum Encodings {
    /// The values of a binary storage column.
    Stored(ScalarColumn),
    /// Encodings that the encoder wrote back to back.
    Written {
        /// The rows that hold a message.
        validity: Mask,
        /// The encodings, back to back.
        bytes: Vec<u8>,
        /// The offset of each encoding into `bytes`, and the end of the last one.
        offsets: Vec<usize>,
    },
}

impl Encodings {
    /// Reads the encodings of a binary storage column.
    fn stored(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Ok(Self::Stored(ScalarColumn::new(
            ScalarKind::Bytes,
            array,
            ctx,
        )?))
    }

    /// Whether `row` holds an encoding.
    fn is_valid(&self, row: usize) -> bool {
        match self {
            Self::Stored(column) => column.validity.value(row),
            Self::Written { validity, .. } => validity.value(row),
        }
    }

    /// The encoding at `row`, which is empty at a null row of written encodings.
    fn get(&self, row: usize) -> &[u8] {
        match self {
            Self::Stored(column) => match column.get(row) {
                ScalarRef::Bytes(bytes) => bytes,
                value => vortex_panic!("Encoded protobuf messages are binary, got {value:?}"),
            },
            Self::Written { bytes, offsets, .. } => &bytes[offsets[row]..offsets[row + 1]],
        }
    }
}

/// Encodes the well-known values of the storage column `array` as messages.
///
/// # Errors
///
/// Returns an error if executing the column fails, or a JSON value is not valid JSON of its type.
fn encode_well_known(
    well_known: WellKnown,
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Encodings> {
    let len = array.len();
    let validity = array.validity()?.execute_mask(len, ctx)?;
    let mut bytes = Vec::new();
    let mut offsets = Vec::with_capacity(len + 1);
    offsets.push(0);
    match well_known {
        WellKnown::Wrapper(kind) => {
            let column = ScalarColumn::new(kind, array, ctx)?;
            for row in 0..len {
                if validity.value(row) {
                    write_wrapper(kind, column.get(row), &mut bytes);
                }
                offsets.push(bytes.len());
            }
        }
        WellKnown::Timestamp => {
            let storage = array
                .execute::<ExtensionArray>(ctx)?
                .storage_array()
                .clone();
            let nanos = primitive_values::<i64>(storage, ctx)?;
            for (row, &nanos) in nanos.iter().enumerate() {
                if validity.value(row) {
                    encode_timestamp(nanos, &mut bytes);
                }
                offsets.push(bytes.len());
            }
        }
        WellKnown::Json(json_type) => {
            let storage = array
                .execute::<ExtensionArray>(ctx)?
                .storage_array()
                .clone();
            let text = ScalarColumn::new(ScalarKind::String, storage, ctx)?;
            for row in 0..len {
                if validity.value(row)
                    && let ScalarRef::Bytes(text) = text.get(row)
                {
                    encode_json(json_type, text, &mut bytes)?;
                }
                offsets.push(bytes.len());
            }
        }
    }
    Ok(Encodings::Written {
        validity,
        bytes,
        offsets,
    })
}

/// The number of bytes of a nested message of `size` bytes, with keys of `key_len` bytes.
fn nested_len(key_len: usize, group: bool, size: usize) -> usize {
    if group {
        2 * key_len + size
    } else {
        key_len + length_delimited_len(size)
    }
}

/// The values of a scalar storage column.
struct ScalarColumn {
    /// The protobuf kind of the values.
    kind: ScalarKind,
    /// The rows that hold a value.
    validity: Mask,
    /// The values, which are undefined at null rows.
    values: ColumnValues,
}

/// The canonical values of a [`ScalarColumn`].
enum ColumnValues {
    /// `f64` values.
    F64(Buffer<f64>),
    /// `f32` values.
    F32(Buffer<f32>),
    /// `i32` values.
    I32(Buffer<i32>),
    /// `i64` values.
    I64(Buffer<i64>),
    /// `u32` values.
    U32(Buffer<u32>),
    /// `u64` values.
    U64(Buffer<u64>),
    /// `bool` values.
    Bool(BitBuffer),
    /// `utf8` or `binary` values.
    Binary(VarBinViewArray),
}

impl ScalarColumn {
    /// Prepares to read values of `kind` from the column `array`.
    fn new(kind: ScalarKind, array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let validity = array.validity()?.execute_mask(array.len(), ctx)?;
        let values = match kind {
            ScalarKind::Double => ColumnValues::F64(primitive_values(array, ctx)?),
            ScalarKind::Float => ColumnValues::F32(primitive_values(array, ctx)?),
            ScalarKind::Int32 | ScalarKind::Sint32 | ScalarKind::Sfixed32 | ScalarKind::Enum => {
                ColumnValues::I32(primitive_values(array, ctx)?)
            }
            ScalarKind::Int64 | ScalarKind::Sint64 | ScalarKind::Sfixed64 => {
                ColumnValues::I64(primitive_values(array, ctx)?)
            }
            ScalarKind::Uint32 | ScalarKind::Fixed32 => {
                ColumnValues::U32(primitive_values(array, ctx)?)
            }
            ScalarKind::Uint64 | ScalarKind::Fixed64 => {
                ColumnValues::U64(primitive_values(array, ctx)?)
            }
            ScalarKind::Bool => {
                ColumnValues::Bool(array.execute::<BoolArray>(ctx)?.into_bit_buffer())
            }
            ScalarKind::String | ScalarKind::Bytes => {
                ColumnValues::Binary(array.execute::<VarBinViewArray>(ctx)?)
            }
        };
        Ok(Self {
            kind,
            validity,
            values,
        })
    }

    /// The value at `row`, which must be valid.
    fn get(&self, row: usize) -> ScalarRef<'_> {
        match &self.values {
            ColumnValues::F64(values) => ScalarRef::F64(values[row]),
            ColumnValues::F32(values) => ScalarRef::F32(values[row]),
            ColumnValues::I32(values) => ScalarRef::I32(values[row]),
            ColumnValues::I64(values) => ScalarRef::I64(values[row]),
            ColumnValues::U32(values) => ScalarRef::U32(values[row]),
            ColumnValues::U64(values) => ScalarRef::U64(values[row]),
            ColumnValues::Bool(values) => ScalarRef::Bool(values.value(row)),
            ColumnValues::Binary(values) => {
                let view = &values.views()[row];
                ScalarRef::Bytes(if view.is_inlined() {
                    view.as_inlined().value()
                } else {
                    let view = view.as_view();
                    &values.buffer(view.buffer_index as usize).as_slice()[view.as_range()]
                })
            }
        }
    }

    /// The value that a singular field writes for `row`, if any: none if the row is null, or if
    /// the field has implicit presence (no `presence`) and the value is its default.
    fn written(&self, row: usize, presence: bool) -> Option<ScalarRef<'_>> {
        if !self.validity.value(row) {
            return None;
        }
        let value = self.get(row);
        (presence || !value.is_default()).then_some(value)
    }

    /// The number of bytes of the values in `rows`, without keys.
    fn payload_len(&self, rows: Range<usize>) -> usize {
        match self.kind.wire_type() {
            WireType::SixtyFourBit => 8 * rows.len(),
            WireType::ThirtyTwoBit => 4 * rows.len(),
            _ => rows.map(|row| self.get(row).encoded_len(self.kind)).sum(),
        }
    }
}

/// The element ranges of the lists in a list column.
struct Lists {
    /// The offset of each list into the elements.
    offsets: Vec<usize>,
    /// The number of elements of each list.
    sizes: Vec<usize>,
}

impl Lists {
    /// The element range of the list at `row`.
    fn range(&self, row: usize) -> Range<usize> {
        let start = self.offsets[row];
        start..start + self.sizes[row]
    }
}

/// Splits a list column into its list ranges and its elements.
fn list_parts(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<(Lists, ArrayRef)> {
    let parts = array.execute::<ListViewArray>(ctx)?.into_data_parts();
    let lists = Lists {
        offsets: integers_as_usize(parts.offsets, ctx)?,
        sizes: integers_as_usize(parts.sizes, ctx)?,
    };
    Ok((lists, parts.elements))
}

/// Executes a primitive array of `T` values and returns its values.
fn primitive_values<T: NativePType>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>> {
    Ok(array.execute::<PrimitiveArray>(ctx)?.into_buffer())
}

/// Executes an integer array and converts its values to `usize`.
fn integers_as_usize(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<usize>> {
    let primitive = array.execute::<PrimitiveArray>(ctx)?;
    Ok(match_each_integer_ptype!(primitive.ptype(), |T| {
        widen(primitive.as_slice::<T>())
    }))
}

/// Converts integers that are known to be non-negative offsets or sizes to `usize`.
fn widen<T: IntegerPType>(values: &[T]) -> Vec<usize> {
    values.iter().map(|value| value.as_()).collect()
}

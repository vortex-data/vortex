// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The decode plan of a message type, compiled into the builders of its storage columns.

use std::cmp::Ordering;

use prost::encoding::DecodeContext;
use prost::encoding::WireType;
use prost::encoding::decode_key;
use prost::encoding::encode_key;
use prost::encoding::encode_varint;
use prost::encoding::skip_field;
use prost_reflect::FieldDescriptor;
use prost_reflect::MessageDescriptor;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use super::builders::Binary;
use super::builders::ListOffsets;
use super::builders::Scalars;
use super::builders::as_usize;
use super::builders::finish_validity;
use super::builders::new_validity;
use super::well_known::WellKnownColumn;
use crate::mapping::FieldPlan;
use crate::mapping::FieldShape;
use crate::mapping::MAP_ENTRY_FIELDS;
use crate::mapping::MessagePlan;
use crate::mapping::UNKNOWN_FIELDS;
use crate::mapping::ValuePlan;
use crate::wire::ClosedEnum;
use crate::wire::ScalarKind;
use crate::wire::consumed;
use crate::wire::read_group;
use crate::wire::read_length_delimited;
use crate::wire::read_varint;
use crate::wire::varint_to_i32;

/// Non-nullable, the nullability of list elements, map keys and map values.
const NON_NULLABLE: Nullability = Nullability::NonNullable;

/// The struct column of a message type, which decodes messages into one column per field.
///
/// Every field column holds at most as many rows as the message column. A field column catches
/// up with absent rows only when a later row sets it, or when the column finishes.
pub(super) struct MessageColumn {
    /// The storage field names: the field names in declaration order, and [`UNKNOWN_FIELDS`] if
    /// the unknown fields are stored.
    names: FieldNames,
    /// The fields, in declaration order.
    fields: Vec<Field>,
    /// The position in `fields` of each field number.
    lookup: FieldLookup,
    /// For each oneof, the row and the position in `fields` of the member that was set last.
    oneofs: Vec<Option<(usize, usize)>>,
    /// The encoded unknown fields of each row, if the mapping stores them.
    unknown: Option<Binary>,
    /// The numbers of a packed run that a closed enum does not define, which become unknown
    /// fields.
    rejected: Vec<i32>,
    /// Whether a field holds well-known values, which are converted once their message is
    /// complete.
    converts: bool,
    /// The validity, or `None` if the column is non-nullable.
    validity: Option<BitBufferMut>,
    /// The number of rows.
    len: usize,
}

impl MessageColumn {
    /// Creates the column of messages that `plan` describes.
    ///
    /// `capacity` is the expected number of rows, which also sizes the fields that hold one value
    /// per row, but not the elements of repeated fields.
    pub(super) fn new(plan: &MessagePlan, nullability: Nullability, capacity: usize) -> Self {
        let oneofs = plan
            .descriptor
            .oneofs()
            .map(|oneof| oneof.name().to_owned())
            .collect::<Vec<_>>();
        let fields = plan
            .fields
            .iter()
            .map(|field| {
                let oneof = field
                    .descriptor
                    .containing_oneof()
                    .and_then(|oneof| oneofs.iter().position(|name| name == oneof.name()));
                Field::new(field, oneof, capacity)
            })
            .collect::<Vec<_>>();
        let names = plan
            .fields
            .iter()
            .map(|field| FieldName::from(field.descriptor.name()))
            .chain(plan.unknown_fields.then(|| FieldName::from(UNKNOWN_FIELDS)))
            .collect();
        Self {
            names,
            lookup: FieldLookup::new(plan.fields.iter().map(|field| field.descriptor.number())),
            oneofs: vec![None; oneofs.len()],
            unknown: plan
                .unknown_fields
                .then(|| Binary::new(DType::Binary(NON_NULLABLE), capacity)),
            rejected: Vec::new(),
            converts: fields.iter().any(|field| field.converts),
            fields,
            validity: new_validity(nullability, capacity),
            len: 0,
        }
    }

    /// The number of rows.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Makes `row` a present message, unless it already is the last row.
    ///
    /// The rows between the last row and `row` become absent: null if the column is nullable, and
    /// empty messages otherwise.
    pub(super) fn enter(&mut self, row: usize) {
        if self.len > row {
            return;
        }
        self.fill(row);
        if let Some(validity) = &mut self.validity {
            validity.append_true();
        }
        self.len += 1;
    }

    /// Appends absent messages until the column holds `len` rows.
    pub(super) fn fill(&mut self, len: usize) {
        if len > self.len {
            if let Some(validity) = &mut self.validity {
                validity.append_n(false, len - self.len);
            }
            self.len = len;
        }
    }

    /// Removes the rows from `len` on, if the column holds more, and the numbers of a packed run
    /// that a failed merge left in `rejected`.
    pub(super) fn truncate(&mut self, len: usize) {
        self.rejected.clear();
        if len >= self.len {
            return;
        }
        for field in &mut self.fields {
            field.column.truncate(len);
        }
        if let Some(unknown) = &mut self.unknown {
            unknown.truncate(len);
        }
        if let Some(validity) = &mut self.validity {
            validity.truncate(len);
        }
        self.len = len;
    }

    /// Decodes fields from `input` into the message at `row`, the last row.
    ///
    /// Without `group`, this reads until `input` is empty. With `group`, this reads until the end
    /// group tag of that field number, which must come before `input` ends.
    pub(super) fn merge(
        &mut self,
        row: usize,
        input: &mut &[u8],
        group: Option<u32>,
    ) -> VortexResult<()> {
        while !input.is_empty() {
            let start = *input;
            let (number, wire_type) = decode_key(input)?;
            if wire_type == WireType::EndGroup {
                if group == Some(number) {
                    return Ok(());
                }
                vortex_bail!("Unexpected protobuf end group tag of field {number}");
            }
            let known = match self.lookup.find(number) {
                // Protobuf parsers treat a value whose wire type does not match its field like an
                // unknown field.
                Some(index) if self.fields[index].accepts(wire_type) => {
                    self.merge_field(row, index, number, wire_type, input)?
                }
                _ => {
                    skip_field(wire_type, number, input, DecodeContext::default())?;
                    false
                }
            };
            if !known && let Some(unknown) = &mut self.unknown {
                unknown.extend(row, consumed(start, input))?;
            }
            if !self.rejected.is_empty() {
                self.reject_numbers(row, number)?;
            }
        }
        if let Some(number) = group {
            vortex_bail!("Truncated protobuf input: group field {number} has no end group tag");
        }
        Ok(())
    }

    /// Decodes one occurrence of the field at `index` into `row`, and returns whether the
    /// occurrence set the field, which a number that a closed enum does not define does not.
    fn merge_field(
        &mut self,
        row: usize,
        index: usize,
        number: u32,
        wire_type: WireType,
        input: &mut &[u8],
    ) -> VortexResult<bool> {
        let field = &mut self.fields[index];
        let set = field
            .column
            .merge(row, number, wire_type, input, &mut self.rejected)?;
        if set && let Some(oneof) = field.oneof {
            // Setting a oneof member clears the member that was set before in the same row.
            if let Some((set_row, set_index)) = self.oneofs[oneof]
                && set_row == row
                && set_index != index
            {
                self.fields[set_index].column.truncate(row);
            }
            self.oneofs[oneof] = Some((row, index));
        }
        Ok(set)
    }

    /// Keeps the numbers of a packed run of field `number` that its closed enum does not define
    /// as unknown varint fields of `row`, as protobuf parsers do, if the mapping stores them.
    fn reject_numbers(&mut self, row: usize, number: u32) -> VortexResult<()> {
        if let Some(unknown) = &mut self.unknown {
            let mut fields = Vec::new();
            for &value in &self.rejected {
                encode_key(number, WireType::Varint, &mut fields);
                encode_varint(i64::from(value).cast_unsigned(), &mut fields);
            }
            unknown.extend(row, &fields)?;
        }
        self.rejected.clear();
        Ok(())
    }

    /// Decodes one occurrence of a message field into `row`, the last row.
    ///
    /// The occurrence is a group if `wire_type` starts one, and length-delimited otherwise.
    fn merge_occurrence(
        &mut self,
        row: usize,
        number: u32,
        wire_type: WireType,
        input: &mut &[u8],
    ) -> VortexResult<()> {
        if wire_type == WireType::StartGroup {
            self.merge(row, input, Some(number))
        } else {
            self.merge(row, &mut read_length_delimited(input)?, None)
        }
    }

    /// Converts the well-known values of the message at `row`, which is complete.
    ///
    /// # Errors
    ///
    /// Returns an error if a well-known value has no stored form, such as a timestamp outside the
    /// range of the storage.
    pub(super) fn convert(&mut self, row: usize) -> VortexResult<()> {
        if !self.converts || row >= self.len {
            return Ok(());
        }
        for field in self.fields.iter_mut().filter(|field| field.converts) {
            field.column.convert(row)?;
        }
        Ok(())
    }

    /// Fills the column to `len` rows and builds its struct array.
    pub(super) fn finish(mut self, len: usize) -> VortexResult<ArrayRef> {
        self.fill(len);
        let mut fields = self
            .fields
            .into_iter()
            .map(|field| field.column.finish(len))
            .collect::<VortexResult<Vec<_>>>()?;
        if let Some(unknown) = self.unknown {
            fields.push(unknown.finish(len));
        }
        Ok(
            StructArray::try_new(self.names, fields, len, finish_validity(self.validity))?
                .into_array(),
        )
    }
}

/// One field of a [`MessageColumn`].
struct Field {
    /// The wire type of a single value of the field.
    wire_type: WireType,
    /// Whether the field also accepts packed values, as a length-delimited run.
    packable: bool,
    /// The index of the oneof that contains the field, if any.
    oneof: Option<usize>,
    /// Whether the field holds well-known values, which are converted once their message is
    /// complete.
    converts: bool,
    /// The storage column of the field.
    column: FieldColumn,
}

impl Field {
    /// Creates the field that `plan` describes, which is a member of the oneof at index `oneof`,
    /// if any.
    fn new(plan: &FieldPlan, oneof: Option<usize>, capacity: usize) -> Self {
        let column = FieldColumn::new(plan, capacity);
        let (wire_type, packable) = match &plan.shape {
            FieldShape::Single(value) => (value_wire_type(&plan.descriptor, value), false),
            FieldShape::List(value) => (
                value_wire_type(&plan.descriptor, value),
                matches!(value, ValuePlan::Scalar { kind, .. } if kind.is_packable()),
            ),
            FieldShape::Map { .. } => (WireType::LengthDelimited, false),
        };
        Self {
            wire_type,
            packable,
            oneof,
            converts: column.converts(),
            column,
        }
    }

    /// Whether the field accepts a value of `wire_type`.
    fn accepts(&self, wire_type: WireType) -> bool {
        wire_type == self.wire_type || (self.packable && wire_type == WireType::LengthDelimited)
    }
}

/// Returns the wire type of a single value of `field` that is stored as `value`.
fn value_wire_type(field: &FieldDescriptor, value: &ValuePlan) -> WireType {
    match value {
        ValuePlan::Scalar { kind, .. } => kind.wire_type(),
        _ if field.is_group() => WireType::StartGroup,
        _ => WireType::LengthDelimited,
    }
}

/// The storage column of a message field.
enum FieldColumn {
    /// A singular field.
    Single(ValueColumn),
    /// A repeated field.
    List {
        /// The lists.
        lists: ListOffsets,
        /// The elements of all lists.
        elements: ValueColumn,
    },
    /// A map field.
    Map(Box<MapColumn>),
}

impl FieldColumn {
    /// Creates the storage column of the field that `plan` describes.
    fn new(plan: &FieldPlan, capacity: usize) -> Self {
        match &plan.shape {
            FieldShape::Single(value) => Self::Single(ValueColumn::new(
                value,
                plan.descriptor.supports_presence().into(),
                capacity,
            )),
            FieldShape::List(value) => Self::List {
                lists: ListOffsets::new(capacity),
                elements: ValueColumn::new(value, NON_NULLABLE, 0),
            },
            FieldShape::Map { entry, key, value } => {
                Self::Map(Box::new(MapColumn::new(entry, *key, value, capacity)))
            }
        }
    }

    /// Whether the column holds well-known values, which are converted once their message is
    /// complete.
    fn converts(&self) -> bool {
        match self {
            Self::Single(values) => values.converts(),
            Self::List { elements, .. } => elements.converts(),
            Self::Map(map) => map.value.converts(),
        }
    }

    /// Merges one occurrence of the field into `row`, the last row of the message column.
    ///
    /// `number` and `wire_type` are those of the field key, and the field accepts `wire_type`.
    /// Returns whether the occurrence set the field, which a value that a closed enum does not
    /// define does not. The numbers of a packed run that a closed enum does not define go to
    /// `rejected`.
    fn merge(
        &mut self,
        row: usize,
        number: u32,
        wire_type: WireType,
        input: &mut &[u8],
        rejected: &mut Vec<i32>,
    ) -> VortexResult<bool> {
        match self {
            Self::Single(values) => values.merge(row, number, wire_type, input),
            Self::List { lists, elements } => {
                lists.open(row, elements.len());
                elements.append(number, wire_type, input, rejected)
            }
            Self::Map(map) => map.merge_entry(row, input),
        }
    }

    /// Removes the rows from `len` on, if the column holds more.
    fn truncate(&mut self, len: usize) {
        match self {
            Self::Single(values) => values.truncate(len),
            Self::List { lists, elements } => {
                if let Some(kept) = lists.truncate(len) {
                    elements.truncate(kept);
                }
            }
            Self::Map(map) => map.truncate(len),
        }
    }

    /// Converts the well-known values of the field at `row`, whose message is complete.
    fn convert(&mut self, row: usize) -> VortexResult<()> {
        match self {
            Self::Single(values) => values.convert(row),
            Self::List { lists, elements } => {
                for element in lists.range(row, elements.len()) {
                    elements.convert(element)?;
                }
                Ok(())
            }
            Self::Map(map) => map.convert(row),
        }
    }

    /// Fills the column to `len` rows and builds its array.
    fn finish(self, len: usize) -> VortexResult<ArrayRef> {
        match self {
            Self::Single(values) => values.finish(len),
            Self::List { lists, elements } => {
                let elements_len = elements.len();
                Ok(list_array(
                    elements.finish(elements_len)?,
                    lists.finish(len, elements_len),
                ))
            }
            Self::Map(map) => map.finish(len),
        }
    }
}

/// The column of the single values of a field: of a singular field, of the elements of a
/// repeated field, or of the keys or values of a map.
enum ValueColumn {
    /// Scalar values.
    Scalar {
        /// The kind of the values.
        kind: ScalarKind,
        /// The numbers that the enum defines, if the values have a closed enum type.
        closed: Option<ClosedEnum>,
        /// The values.
        values: Scalars,
    },
    /// Messages, as structs.
    Message(Box<MessageColumn>),
    /// Messages beyond the recursion depth, as their encodings.
    Encoded(Binary),
    /// Well-known messages, in their dedicated forms.
    WellKnown(Box<WellKnownColumn>),
}

impl ValueColumn {
    /// Creates the column of values that `plan` describes.
    fn new(plan: &ValuePlan, nullability: Nullability, capacity: usize) -> Self {
        match plan {
            ValuePlan::Scalar { kind, enum_type } => Self::Scalar {
                kind: *kind,
                closed: enum_type.as_ref().and_then(ClosedEnum::of),
                // Absent enum values hold the first number of the enum, its default.
                values: Scalars::new(
                    *kind,
                    nullability,
                    enum_type
                        .as_ref()
                        .map_or(0, |enum_type| enum_type.default_value().number()),
                    capacity,
                ),
            },
            ValuePlan::Message(message) => {
                Self::Message(Box::new(MessageColumn::new(message, nullability, capacity)))
            }
            ValuePlan::Encoded => Self::Encoded(Binary::new(DType::Binary(nullability), capacity)),
            ValuePlan::WellKnown(well_known) => Self::WellKnown(Box::new(WellKnownColumn::new(
                *well_known,
                nullability,
                capacity,
            ))),
        }
    }

    /// The number of rows.
    fn len(&self) -> usize {
        match self {
            Self::Scalar { values, .. } => values.len(),
            Self::Message(column) => column.len(),
            Self::Encoded(values) => values.len(),
            Self::WellKnown(column) => column.len(),
        }
    }

    /// Whether the column holds well-known values, directly or in its messages.
    fn converts(&self) -> bool {
        match self {
            Self::Scalar { .. } | Self::Encoded(_) => false,
            Self::Message(column) => column.converts,
            Self::WellKnown(_) => true,
        }
    }

    /// Merges one occurrence of a singular value into `row`, which is the last row or the one
    /// after it.
    ///
    /// The last scalar wins, and a message merges into the message that is already there.
    /// Returns false, leaving the row as it was, for a number that a closed enum does not define.
    fn merge(
        &mut self,
        row: usize,
        number: u32,
        wire_type: WireType,
        input: &mut &[u8],
    ) -> VortexResult<bool> {
        match self {
            Self::Scalar {
                kind,
                closed,
                values,
            } => {
                if let Some(closed) = closed {
                    let mut peek = *input;
                    if !closed.contains(varint_to_i32(read_varint(&mut peek)?)) {
                        *input = peek;
                        return Ok(false);
                    }
                }
                values.truncate(row);
                values.fill(row);
                values.append_from(*kind, input, false)?;
            }
            Self::Message(column) => {
                column.enter(row);
                column.merge_occurrence(row, number, wire_type, input)?;
            }
            // The encodings of two occurrences, one after the other, merge as the messages do.
            Self::Encoded(values) => values.extend(row, read_message(number, wire_type, input)?)?,
            Self::WellKnown(column) => column.extend(row, read_length_delimited(input)?)?,
        }
        Ok(true)
    }

    /// Appends the values of one occurrence of a repeated field.
    ///
    /// A packed run of scalars appends all its values, and drops the numbers that a closed enum
    /// does not define to `rejected`. Returns false, appending nothing, for a single number that
    /// a closed enum does not define.
    fn append(
        &mut self,
        number: u32,
        wire_type: WireType,
        input: &mut &[u8],
        rejected: &mut Vec<i32>,
    ) -> VortexResult<bool> {
        let row = self.len();
        match self {
            // A length-delimited occurrence of a numeric kind is a packed run of values.
            Self::Scalar {
                kind,
                closed,
                values,
            } if wire_type != kind.wire_type() => {
                let mut packed = read_length_delimited(input)?;
                match closed {
                    Some(closed) => values.append_closed_enums(closed, &mut packed, rejected)?,
                    None => values.append_from(*kind, &mut packed, true)?,
                }
                Ok(true)
            }
            _ => self.merge(row, number, wire_type, input),
        }
    }

    /// Appends absent rows until the column holds `len` rows.
    fn fill(&mut self, len: usize) {
        match self {
            Self::Scalar { values, .. } => values.fill(len),
            Self::Message(column) => column.fill(len),
            Self::Encoded(values) => values.fill(len),
            Self::WellKnown(column) => column.fill(len),
        }
    }

    /// Removes the rows from `len` on, if the column holds more.
    fn truncate(&mut self, len: usize) {
        match self {
            Self::Scalar { values, .. } => values.truncate(len),
            Self::Message(column) => column.truncate(len),
            Self::Encoded(values) => values.truncate(len),
            Self::WellKnown(column) => column.truncate(len),
        }
    }

    /// Converts the well-known values at `row`, whose message is complete.
    fn convert(&mut self, row: usize) -> VortexResult<()> {
        match self {
            Self::Scalar { .. } | Self::Encoded(_) => Ok(()),
            Self::Message(column) => column.convert(row),
            Self::WellKnown(column) => column.convert(row),
        }
    }

    /// Fills the column to `len` rows and builds its array.
    fn finish(self, len: usize) -> VortexResult<ArrayRef> {
        match self {
            Self::Scalar { values, .. } => Ok(values.finish(len)),
            Self::Message(column) => column.finish(len),
            Self::Encoded(values) => Ok(values.finish(len)),
            Self::WellKnown(column) => column.finish(len),
        }
    }

    /// Compares the scalar values at rows `a` and `b`, which must both be present.
    fn compare(&self, a: usize, b: usize) -> Ordering {
        match self {
            Self::Scalar { values, .. } => values.compare(a, b),
            _ => vortex_panic!("Protobuf map keys are scalars"),
        }
    }
}

/// Reads the fields of one occurrence of a message field: a group if `wire_type` starts one, and
/// a length-delimited message otherwise.
fn read_message<'a>(
    number: u32,
    wire_type: WireType,
    input: &mut &'a [u8],
) -> VortexResult<&'a [u8]> {
    if wire_type == WireType::StartGroup {
        read_group(input, number)
    } else {
        read_length_delimited(input)
    }
}

/// The list column of a map field, whose elements are `{key, value}` entries.
///
/// Entries are appended in the order they are read. A row whose keys are not strictly increasing
/// is recorded, and [`finish`](Self::finish) sorts its entries by key and keeps the last entry of
/// each key.
struct MapColumn {
    /// The lists of entries.
    lists: ListOffsets,
    /// The keys of all entries.
    key: ValueColumn,
    /// The values of all entries.
    value: ValueColumn,
    /// The wire types of the key and the value field of the entry message.
    wire_types: [WireType; 2],
    /// The number of entries of all maps.
    entries: usize,
    /// The rows whose keys are not strictly increasing, in increasing order.
    unordered: Vec<usize>,
}

impl MapColumn {
    /// Creates the column of a map field with the entry message `entry`, whose keys are of `key`
    /// kind and whose values `value` describes.
    fn new(entry: &MessageDescriptor, key: ScalarKind, value: &ValuePlan, capacity: usize) -> Self {
        let key = ValuePlan::Scalar {
            kind: key,
            enum_type: None,
        };
        Self {
            lists: ListOffsets::new(capacity),
            wire_types: [
                value_wire_type(&entry.map_entry_key_field(), &key),
                value_wire_type(&entry.map_entry_value_field(), value),
            ],
            key: ValueColumn::new(&key, NON_NULLABLE, 0),
            value: ValueColumn::new(value, NON_NULLABLE, 0),
            entries: 0,
            unordered: Vec::new(),
        }
    }

    /// Decodes one length-delimited entry from `input` into the map at `row`.
    ///
    /// A missing key or value takes its default. Returns false and drops the entry if its value
    /// is a number that a closed enum does not define, as protobuf parsers do.
    fn merge_entry(&mut self, row: usize, input: &mut &[u8]) -> VortexResult<bool> {
        let mut payload = read_length_delimited(input)?;
        let entry = self.entries;
        self.lists.open(row, entry);
        let mut dropped = false;
        while !payload.is_empty() {
            let (number, wire_type) = decode_key(&mut payload)?;
            let column = match number {
                1 if wire_type == self.wire_types[0] => &mut self.key,
                2 if wire_type == self.wire_types[1] => &mut self.value,
                _ => {
                    skip_field(wire_type, number, &mut payload, DecodeContext::default())?;
                    continue;
                }
            };
            let set = column.merge(entry, number, wire_type, &mut payload)?;
            if number == 2 {
                dropped = !set;
            }
        }
        if dropped {
            self.key.truncate(entry);
            self.value.truncate(entry);
            return Ok(false);
        }

        self.entries += 1;
        self.key.fill(self.entries);
        self.value.fill(self.entries);
        if entry > self.lists.open_start()
            && self.key.compare(entry - 1, entry) != Ordering::Less
            && self.unordered.last() != Some(&row)
        {
            self.unordered.push(row);
        }
        Ok(true)
    }

    /// Removes the rows from `len` on, if the column holds more.
    fn truncate(&mut self, len: usize) {
        if let Some(kept) = self.lists.truncate(len) {
            self.key.truncate(kept);
            self.value.truncate(kept);
            self.entries = kept;
        }
        while self.unordered.last().is_some_and(|&row| row >= len) {
            self.unordered.pop();
        }
    }

    /// Converts the well-known values of the map at `row`, whose message is complete.
    ///
    /// Only the entries that [`finish`](Self::finish) keeps convert, so a value that a later entry
    /// with the same key replaces need not have a stored form. The values of the replaced entries
    /// are filled with defaults, which `finish` drops.
    fn convert(&mut self, row: usize) -> VortexResult<()> {
        let entries = self.lists.range(row, self.entries);
        if self.unordered.binary_search(&row).is_err() {
            for entry in entries {
                self.value.convert(entry)?;
            }
            return Ok(());
        }
        let mut kept = entries.collect::<Vec<_>>();
        sort_keeping_last(&mut kept, |a, b| self.key.compare(a, b));
        // The values convert in entry order, which their columns append in.
        kept.sort_unstable();
        for entry in kept {
            self.value.convert(entry)?;
        }
        Ok(())
    }

    /// Fills the column to `len` rows and builds its list array, with the entries of each map
    /// ordered by key.
    fn finish(self, len: usize) -> VortexResult<ArrayRef> {
        let Self {
            lists,
            key,
            value,
            entries,
            unordered,
            ..
        } = self;
        let offsets = lists.finish(len, entries);
        let order = (!unordered.is_empty())
            .then(|| sorted_entries(&offsets, &unordered, |a, b| key.compare(a, b)));
        let mut entries = StructArray::try_new(
            FieldNames::from(MAP_ENTRY_FIELDS),
            [key.finish(entries)?, value.finish(entries)?],
            entries,
            Validity::NonNullable,
        )?
        .into_array();
        let offsets = match order {
            Some((indices, offsets)) => {
                entries = entries.take(indices.into_array())?;
                offsets
            }
            None => offsets,
        };
        Ok(list_array(entries, offsets))
    }
}

/// Orders the entries of the `unordered` rows of a map column by key, keeping the last entry of
/// each key, and leaves the other rows as they are.
///
/// Returns the indices of the kept entries in their new order, and the offsets of the lists into
/// them.
fn sorted_entries(
    offsets: &[u64],
    unordered: &[usize],
    compare: impl Fn(usize, usize) -> Ordering,
) -> (Buffer<u64>, Buffer<u64>) {
    let total = offsets.last().copied().unwrap_or_default();
    let mut indices = BufferMut::<u64>::with_capacity(as_usize(total));
    let mut new_offsets = BufferMut::<u64>::with_capacity(offsets.len());
    new_offsets.push(0);
    let mut unordered = unordered.iter().copied().peekable();
    let mut row_entries = Vec::new();
    for (row, bounds) in offsets.windows(2).enumerate() {
        let range = bounds[0]..bounds[1];
        if unordered.next_if_eq(&row).is_some() {
            row_entries.clear();
            row_entries.extend(as_usize(range.start)..as_usize(range.end));
            sort_keeping_last(&mut row_entries, &compare);
            indices.extend(row_entries.iter().map(|&entry| entry as u64));
        } else {
            indices.extend(range);
        }
        new_offsets.push(indices.len() as u64);
    }
    (indices.freeze(), new_offsets.freeze())
}

/// Sorts the entries of one map by key, and keeps only the last entry of each key, which replaces
/// the ones read before it.
fn sort_keeping_last(entries: &mut Vec<usize>, compare: impl Fn(usize, usize) -> Ordering) {
    // A stable sort keeps the entries of one key in the order they were read.
    entries.sort_by(|&a, &b| compare(a, b));
    // `dedup_by` keeps the first entry of a run of equal keys, so that slot takes the later entry.
    entries.dedup_by(|later, kept| {
        let same = compare(*later, *kept) == Ordering::Equal;
        if same {
            *kept = *later;
        }
        same
    });
}

/// Builds a non-nullable list array from its elements and offsets.
fn list_array(elements: ArrayRef, offsets: Buffer<u64>) -> ArrayRef {
    // SAFETY: the offsets start at zero and never decrease, because each list starts where the
    // one before it ends, and the last one is the number of elements. The lists are non-nullable,
    // as `Validity::NonNullable` states.
    unsafe { ListArray::new_unchecked(elements, offsets.into_array(), Validity::NonNullable) }
        .into_array()
}

/// Finds the position of a field by its number.
enum FieldLookup {
    /// A table indexed by field number, for messages whose field numbers are small.
    Dense(Box<[u32]>),
    /// Field numbers with their positions, sorted by number.
    Sparse(Box<[(u32, u32)]>),
}

impl FieldLookup {
    /// The largest field number that a dense table indexes.
    const MAX_DENSE_NUMBER: u32 = 1024;

    /// Marks a number without a field in a dense table.
    const NO_FIELD: u32 = u32::MAX;

    /// Creates the lookup of fields that have the given numbers, in order.
    fn new(numbers: impl Iterator<Item = u32>) -> Self {
        let mut fields = numbers
            .enumerate()
            .map(|(position, number)| {
                let position = u32::try_from(position).vortex_expect("field count fits in u32");
                (number, position)
            })
            .collect::<Vec<_>>();
        let max_number = fields.iter().map(|&(number, _)| number).max();
        match max_number {
            Some(max_number) if max_number > Self::MAX_DENSE_NUMBER => {
                fields.sort_unstable();
                Self::Sparse(fields.into_boxed_slice())
            }
            _ => {
                let mut table = vec![Self::NO_FIELD; max_number.map_or(0, |max| max as usize + 1)];
                for (number, position) in fields {
                    table[number as usize] = position;
                }
                Self::Dense(table.into_boxed_slice())
            }
        }
    }

    /// Returns the position of the field `number`, or `None` if the message has no such field.
    fn find(&self, number: u32) -> Option<usize> {
        let position = match self {
            Self::Dense(table) => *table
                .get(number as usize)
                .filter(|&&position| position != Self::NO_FIELD)?,
            Self::Sparse(fields) => {
                let index = fields
                    .binary_search_by_key(&number, |&(number, _)| number)
                    .ok()?;
                fields[index].1
            }
        };
        Some(position as usize)
    }
}

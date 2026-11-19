// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Seeded random protobuf messages, and non-canonical encodings of them.
//!
//! The generator writes the wire format itself, so its encodings do not depend on the encoders
//! under test. A non-canonical encoding parses to the same message as the canonical one, but
//! exercises the parse rules: fields in any order, repeated singular fields whose last value wins,
//! oneof members that replace each other, message fields split into occurrences that merge,
//! packed and unpacked runs of one repeated field, map entries with duplicate keys or missing keys
//! and values, overlong varints and unknown fields.

use std::collections::VecDeque;
use std::iter::repeat_n;
use std::mem;

use prost::bytes::Bytes;
use prost::encoding::WireType;
use prost::encoding::encode_varint;
use prost_reflect::DynamicMessage;
use prost_reflect::FieldDescriptor;
use prost_reflect::Kind;
use prost_reflect::MapKey;
use prost_reflect::MessageDescriptor;
use prost_reflect::ReflectMessage;
use prost_reflect::Syntax;
use prost_reflect::Value;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::seq::SliceRandom;

/// Characters of random strings, which cover one to four UTF-8 bytes.
const CHARS: [char; 8] = ['a', 'Z', '0', ' ', 'é', 'ß', '日', '😀'];

/// The nanoseconds of a second.
const NANOS_PER_SECOND: i64 = 1_000_000_000;

/// Generates random messages and their encodings, deterministically from a seed.
pub(super) struct Generator {
    /// The source of all random choices.
    rng: StdRng,
    /// Whether well-known messages hold only values that their dedicated forms store, see
    /// [`Self::for_dedicated_forms`].
    dedicated_forms: bool,
}

impl Generator {
    /// Creates a generator whose choices follow from `seed`.
    pub(super) fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            dedicated_forms: false,
        }
    }

    /// Makes the messages hold only values that the dedicated forms of the well-known types store:
    /// timestamps within the `i64` nanosecond range, and JSON values that have a kind, a finite
    /// number and `NULL_VALUE` as their null, the only number that JSON `null` keeps.
    ///
    /// The values that a non-canonical encoding writes before the value that replaces them may
    /// still be any value.
    pub(super) fn for_dedicated_forms(mut self) -> Self {
        self.dedicated_forms = true;
        self
    }

    /// Returns a random number below `bound`.
    pub(super) fn below(&mut self, bound: usize) -> usize {
        self.rng.random_range(0..bound)
    }

    /// Returns true with probability `p`.
    pub(super) fn chance(&mut self, p: f64) -> bool {
        self.rng.random_bool(p)
    }

    /// Returns a random message of type `descriptor`.
    ///
    /// Each field is set with some probability, with values that favor edge cases. A oneof gets
    /// at most one member. A closed enum field only holds numbers that the enum defines.
    pub(super) fn message(&mut self, descriptor: &MessageDescriptor) -> DynamicMessage {
        let mut message = DynamicMessage::new(descriptor.clone());
        for field in descriptor.fields() {
            let in_oneof = field
                .containing_oneof()
                .is_some_and(|oneof| !oneof.is_synthetic());
            if !in_oneof && self.chance(0.6) {
                let value = self.field_value(&field);
                message.set_field(&field, value);
            }
        }
        for oneof in descriptor.oneofs().filter(|oneof| !oneof.is_synthetic()) {
            let members = oneof.fields().collect::<Vec<_>>();
            if self.chance(0.7)
                && let Some(member) = members.choose(&mut self.rng)
            {
                let value = self.value(&member.kind());
                message.set_field(member, value);
            }
        }
        if self.dedicated_forms {
            self.make_storable(&mut message);
        }
        message
    }

    /// Replaces the values of a well-known `message` that its dedicated form does not store.
    fn make_storable(&mut self, message: &mut DynamicMessage) {
        match message.descriptor().full_name() {
            "google.protobuf.Timestamp" => {
                let random = self.rng.random();
                let nanos = self.edge(&[i64::MIN, i64::MAX, -1, 0], random);
                let subsecond = i32::try_from(nanos.rem_euclid(NANOS_PER_SECOND))
                    .unwrap_or_else(|_| unreachable!());
                message
                    .set_field_by_name("seconds", Value::I64(nanos.div_euclid(NANOS_PER_SECOND)));
                message.set_field_by_name("nanos", Value::I32(subsecond));
            }
            "google.protobuf.Value" => {
                let finite = message
                    .get_field_by_name("number_value")
                    .and_then(|number| number.as_f64())
                    .is_some_and(f64::is_finite);
                if message.has_field_by_name("null_value") {
                    message.set_field_by_name("null_value", Value::EnumNumber(0));
                } else if message.has_field_by_name("number_value") && !finite {
                    let number = self.rng.random_range(-1e6..1e6);
                    message.set_field_by_name("number_value", Value::F64(number));
                } else if !message
                    .descriptor()
                    .fields()
                    .any(|field| message.has_field(&field))
                {
                    let value = self.chance(0.5);
                    message.set_field_by_name("bool_value", Value::Bool(value));
                }
            }
            _ => {}
        }
    }

    /// Returns a random value of `kind` that a later value replaces, so that it need not have a
    /// stored form.
    fn decoy(&mut self, kind: &Kind) -> Value {
        let dedicated_forms = mem::replace(&mut self.dedicated_forms, false);
        let decoy = self.value(kind);
        self.dedicated_forms = dedicated_forms;
        decoy
    }

    /// Returns a random value of `field`: a list, a map or a single value.
    fn field_value(&mut self, field: &FieldDescriptor) -> Value {
        let kind = field.kind();
        if field.is_map() {
            let entry = kind.as_message().cloned().unwrap_or_else(|| unreachable!());
            let (key, value) = (entry.map_entry_key_field(), entry.map_entry_value_field());
            let len = self.below(4);
            Value::Map(
                (0..len)
                    .map(|_| {
                        let mut entry_value = self.value(&value.kind());
                        // `prost-reflect` does not write a map value without presence that
                        // equals its default, so the canonical encoding would lose a `-0.0`.
                        if !value.supports_presence() && entry_value == value.default_value() {
                            entry_value = value.default_value();
                        }
                        (self.map_key(&key.kind()), entry_value)
                    })
                    .collect(),
            )
        } else if field.is_list() {
            let len = self.below(5);
            Value::List((0..len).map(|_| self.value(&kind)).collect())
        } else {
            self.value(&kind)
        }
    }

    /// Returns a random single value of `kind`.
    pub(super) fn value(&mut self, kind: &Kind) -> Value {
        match kind {
            Kind::Double => {
                let random = self.rng.random_range(-1e6..1e6);
                Value::F64(self.edge(
                    &[
                        0.0,
                        -0.0,
                        1.0,
                        -1.5,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::MIN_POSITIVE,
                        f64::from_bits(1),
                        f64::MAX,
                    ],
                    random,
                ))
            }
            Kind::Float => {
                let random = self.rng.random_range(-1e6..1e6);
                Value::F32(self.edge(
                    &[
                        0.0,
                        -0.0,
                        1.0,
                        f32::INFINITY,
                        f32::NEG_INFINITY,
                        f32::from_bits(1),
                        f32::MAX,
                    ],
                    random,
                ))
            }
            Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 => {
                let random = self.rng.random();
                Value::I32(self.edge(&[0, 1, -1, i32::MIN, i32::MAX], random))
            }
            Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 => {
                let random = self.rng.random();
                Value::I64(self.edge(&[0, 1, -1, i64::MIN, i64::MAX], random))
            }
            Kind::Uint32 | Kind::Fixed32 => {
                let random = self.rng.random();
                Value::U32(self.edge(&[0, 1, 127, 128, u32::MAX], random))
            }
            Kind::Uint64 | Kind::Fixed64 => {
                let random = self.rng.random();
                Value::U64(self.edge(&[0, 1, 127, 128, u64::MAX], random))
            }
            Kind::Bool => Value::Bool(self.chance(0.5)),
            Kind::String => Value::String(self.string()),
            Kind::Bytes => {
                let len = self.below(20);
                Value::Bytes(Bytes::from(
                    (0..len).map(|_| self.rng.random()).collect::<Vec<u8>>(),
                ))
            }
            Kind::Enum(enum_type) => {
                let defined = enum_type
                    .values()
                    .map(|value| value.number())
                    .collect::<Vec<_>>();
                let closed = enum_type.parent_file().syntax() == Syntax::Proto2;
                let number = if closed || self.chance(0.7) {
                    *defined.choose(&mut self.rng).unwrap_or(&0)
                } else {
                    self.rng.random_range(-3..100)
                };
                Value::EnumNumber(number)
            }
            Kind::Message(message) => Value::Message(self.message(message)),
        }
    }

    /// Returns a random map key of `kind`.
    fn map_key(&mut self, kind: &Kind) -> MapKey {
        match self.value(kind) {
            Value::Bool(key) => MapKey::Bool(key),
            Value::I32(key) => MapKey::I32(key),
            Value::I64(key) => MapKey::I64(key),
            Value::U32(key) => MapKey::U32(key),
            Value::U64(key) => MapKey::U64(key),
            Value::String(key) => MapKey::String(key),
            value => unreachable!("{value:?} cannot be a map key"),
        }
    }

    /// Returns one of `edges`, or `random`.
    fn edge<T: Copy>(&mut self, edges: &[T], random: T) -> T {
        if self.chance(0.4) {
            edges[self.below(edges.len())]
        } else {
            random
        }
    }

    /// Returns a random string, sometimes too long to be inlined in a Vortex view.
    fn string(&mut self) -> String {
        let len = self.below(18);
        (0..len).map(|_| CHARS[self.below(CHARS.len())]).collect()
    }

    /// Encodes `message` with random choices that do not change the message it parses to.
    pub(super) fn non_canonical(&mut self, message: &DynamicMessage) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_message(message, true, &mut out);
        out
    }

    /// Writes the fields of `message`, interleaving the occurrences of different fields.
    ///
    /// A `whole` message may write a field that it does not set, as long as the field ends up
    /// unset. A part of a split message may not, because the other part may set the field.
    fn write_message(&mut self, message: &DynamicMessage, whole: bool, out: &mut Vec<u8>) {
        let descriptor = message.descriptor();
        let mut sequences = Vec::new();
        for field in descriptor.fields() {
            let mut chunks = Vec::new();
            if message.has_field(&field) {
                self.write_decoys(&field, &mut chunks);
                self.write_field(&field, &message.get_field(&field), &mut chunks);
            } else if whole
                && is_singular_scalar(&field)
                && !field.supports_presence()
                && self.chance(0.1)
            {
                // A value followed by the default value leaves the default.
                let decoy = self.decoy(&field.kind());
                self.write_field(&field, &decoy, &mut chunks);
                self.write_field(&field, &field.default_value(), &mut chunks);
            } else if field.is_list() && is_packable(&field.kind()) && self.chance(0.1) {
                // An empty packed run adds no elements.
                let mut chunk = Vec::new();
                self.write_key(field.number(), WireType::LengthDelimited, &mut chunk);
                self.write_varint(0, &mut chunk);
                chunks.push(chunk);
            }
            if !chunks.is_empty() {
                sequences.push(VecDeque::from(chunks));
            }
        }
        if self.chance(0.3) {
            let number = loop {
                let number = self.rng.random_range(1..2000);
                if descriptor.get_field(number).is_none() {
                    break number;
                }
            };
            let mut chunk = Vec::new();
            self.write_unknown(number, 0, &mut chunk);
            sequences.push(VecDeque::from([chunk]));
        }
        while !sequences.is_empty() {
            let index = self.below(sequences.len());
            if let Some(chunk) = sequences[index].pop_front() {
                out.extend_from_slice(&chunk);
            }
            if sequences[index].is_empty() {
                sequences.swap_remove(index);
            }
        }
    }

    /// Writes occurrences before the real value of `field` that the real value replaces: other
    /// values of a singular scalar field, or members of the field's oneof.
    fn write_decoys(&mut self, field: &FieldDescriptor, chunks: &mut Vec<Vec<u8>>) {
        if !self.chance(0.3) {
            return;
        }
        match field
            .containing_oneof()
            .filter(|oneof| !oneof.is_synthetic())
        {
            Some(oneof) => {
                let members = oneof.fields().collect::<Vec<_>>();
                let is_message = field.kind().as_message().is_some();
                let count = 1 + self.below(2);
                for position in 0..count {
                    let last = position + 1 == count;
                    // The real value of a message member would merge into a decoy of the same
                    // member right before it, so that decoy must be another member.
                    let candidates = members
                        .iter()
                        .filter(|&member| !(last && is_message && member == field))
                        .collect::<Vec<_>>();
                    let Some(&member) = candidates.choose(&mut self.rng) else {
                        return;
                    };
                    let decoy = self.decoy(&member.kind());
                    self.write_field(member, &decoy, chunks);
                }
            }
            None if is_singular_scalar(field) => {
                let decoy = self.decoy(&field.kind());
                self.write_field(field, &decoy, chunks);
            }
            None => {}
        }
    }

    /// Writes the occurrences of `field` with `value`, each as one chunk.
    fn write_field(&mut self, field: &FieldDescriptor, value: &Value, chunks: &mut Vec<Vec<u8>>) {
        let kind = field.kind();
        match value {
            Value::Map(entries) => {
                // Sorting first keeps the order of the hash map out of the random choices.
                let mut entries = entries.iter().collect::<Vec<_>>();
                entries.sort_by_key(|&(key, _)| key);
                entries.shuffle(&mut self.rng);
                let entry_type = kind.as_message().cloned().unwrap_or_else(|| unreachable!());
                for (key, value) in entries {
                    if self.chance(0.2) {
                        // An earlier entry with the same key is replaced.
                        let decoy = self.decoy(&entry_type.map_entry_value_field().kind());
                        chunks.push(self.map_entry(field, key, &decoy));
                    }
                    chunks.push(self.map_entry(field, key, value));
                }
            }
            Value::List(values) if is_packable(&kind) => {
                let mut rest = values.as_slice();
                while !rest.is_empty() {
                    let (run, tail) = rest.split_at(1 + self.below(rest.len()));
                    if self.chance(0.5) {
                        let mut payload = Vec::new();
                        for value in run {
                            self.write_scalar(&kind, value, &mut payload);
                        }
                        let mut chunk = Vec::new();
                        self.write_key(field.number(), WireType::LengthDelimited, &mut chunk);
                        self.write_length_delimited(&payload, &mut chunk);
                        chunks.push(chunk);
                    } else {
                        for value in run {
                            chunks.push(self.occurrence(field, value));
                        }
                    }
                    rest = tail;
                }
            }
            Value::List(values) => {
                for value in values {
                    chunks.push(self.occurrence(field, value));
                }
            }
            Value::Message(message) if self.chance(0.3) => {
                // Occurrences of a message field merge.
                let (first, second) = self.split(message);
                chunks.push(self.message_occurrence(field, &first, false));
                chunks.push(self.message_occurrence(field, &second, false));
            }
            value => chunks.push(self.occurrence(field, value)),
        }
    }

    /// Splits `message` into two messages whose merge is `message`.
    fn split(&mut self, message: &DynamicMessage) -> (DynamicMessage, DynamicMessage) {
        let mut first = DynamicMessage::new(message.descriptor());
        let mut second = DynamicMessage::new(message.descriptor());
        for (field, value) in message.fields() {
            let part = if self.chance(0.5) {
                &mut first
            } else {
                &mut second
            };
            part.set_field(&field, value.clone());
        }
        (first, second)
    }

    /// Writes one occurrence of `field` with a single `value`: a scalar, a message or a group.
    fn occurrence(&mut self, field: &FieldDescriptor, value: &Value) -> Vec<u8> {
        if let Value::Message(message) = value {
            return self.message_occurrence(field, message, true);
        }
        let kind = field.kind();
        let mut chunk = Vec::new();
        self.write_key(field.number(), kind.wire_type(), &mut chunk);
        self.write_scalar(&kind, value, &mut chunk);
        chunk
    }

    /// Writes one occurrence of the message or group `field` with `message`, which is a `whole`
    /// message or a part of one.
    fn message_occurrence(
        &mut self,
        field: &FieldDescriptor,
        message: &DynamicMessage,
        whole: bool,
    ) -> Vec<u8> {
        let mut chunk = Vec::new();
        if field.is_group() {
            self.write_key(field.number(), WireType::StartGroup, &mut chunk);
            self.write_message(message, whole, &mut chunk);
            self.write_key(field.number(), WireType::EndGroup, &mut chunk);
        } else {
            let mut payload = Vec::new();
            self.write_message(message, whole, &mut payload);
            self.write_key(field.number(), WireType::LengthDelimited, &mut chunk);
            self.write_length_delimited(&payload, &mut chunk);
        }
        chunk
    }

    /// Writes one map entry of `field`, with its key and value in random order. A key or value
    /// that holds its default may be left out, and an unknown field may be added.
    fn map_entry(&mut self, field: &FieldDescriptor, key: &MapKey, value: &Value) -> Vec<u8> {
        let entry_type = field
            .kind()
            .as_message()
            .cloned()
            .unwrap_or_else(|| unreachable!());
        let (key_field, value_field) = (
            entry_type.map_entry_key_field(),
            entry_type.map_entry_value_field(),
        );
        let key = Value::from(key.clone());
        let mut parts = Vec::new();
        if !(is_default(&key, &key_field.default_value()) && self.chance(0.5)) {
            parts.push(self.occurrence(&key_field, &key));
        }
        if !(is_default(value, &value_field.default_value()) && self.chance(0.5)) {
            parts.push(self.occurrence(&value_field, value));
        }
        if self.chance(0.2) {
            let mut unknown = Vec::new();
            let number = self.rng.random_range(3..10);
            self.write_unknown(number, 0, &mut unknown);
            parts.push(unknown);
        }
        parts.shuffle(&mut self.rng);
        let mut chunk = Vec::new();
        self.write_key(field.number(), WireType::LengthDelimited, &mut chunk);
        self.write_length_delimited(&parts.concat(), &mut chunk);
        chunk
    }

    /// Writes an unknown field `number` with a random value, which is a group with unknown fields
    /// up to a nesting depth of two.
    fn write_unknown(&mut self, number: u32, depth: usize, out: &mut Vec<u8>) {
        match self.below(if depth < 2 { 5 } else { 4 }) {
            0 => {
                self.write_key(number, WireType::Varint, out);
                let value = self.rng.random();
                self.write_varint(value, out);
            }
            1 => {
                self.write_key(number, WireType::ThirtyTwoBit, out);
                out.extend_from_slice(&self.rng.random::<u32>().to_le_bytes());
            }
            2 => {
                self.write_key(number, WireType::SixtyFourBit, out);
                out.extend_from_slice(&self.rng.random::<u64>().to_le_bytes());
            }
            3 => {
                self.write_key(number, WireType::LengthDelimited, out);
                let len = self.below(8);
                let payload = (0..len).map(|_| self.rng.random()).collect::<Vec<u8>>();
                self.write_length_delimited(&payload, out);
            }
            _ => {
                self.write_key(number, WireType::StartGroup, out);
                for _ in 0..self.below(3) {
                    let inner = self.rng.random_range(1..2000);
                    self.write_unknown(inner, depth + 1, out);
                }
                self.write_key(number, WireType::EndGroup, out);
            }
        }
    }

    /// Writes a scalar `value` of `kind` without a key.
    fn write_scalar(&mut self, kind: &Kind, value: &Value, out: &mut Vec<u8>) {
        match (kind, value) {
            (Kind::Double, Value::F64(value)) => out.extend_from_slice(&value.to_le_bytes()),
            (Kind::Float, Value::F32(value)) => out.extend_from_slice(&value.to_le_bytes()),
            (Kind::Fixed32, Value::U32(value)) => out.extend_from_slice(&value.to_le_bytes()),
            (Kind::Fixed64, Value::U64(value)) => out.extend_from_slice(&value.to_le_bytes()),
            (Kind::Sfixed32, Value::I32(value)) => out.extend_from_slice(&value.to_le_bytes()),
            (Kind::Sfixed64, Value::I64(value)) => out.extend_from_slice(&value.to_le_bytes()),
            (Kind::Int32, Value::I32(value)) | (Kind::Enum(_), Value::EnumNumber(value)) => {
                self.write_varint(i64::from(*value).cast_unsigned(), out)
            }
            (Kind::Int64, Value::I64(value)) => self.write_varint(value.cast_unsigned(), out),
            (Kind::Uint32, Value::U32(value)) => self.write_varint(u64::from(*value), out),
            (Kind::Uint64, Value::U64(value)) => self.write_varint(*value, out),
            (Kind::Sint32, Value::I32(value)) => {
                let zigzag = ((value << 1) ^ (value >> 31)).cast_unsigned();
                self.write_varint(u64::from(zigzag), out)
            }
            (Kind::Sint64, Value::I64(value)) => {
                self.write_varint(((value << 1) ^ (value >> 63)).cast_unsigned(), out)
            }
            (Kind::Bool, Value::Bool(value)) => self.write_varint(u64::from(*value), out),
            (Kind::String, Value::String(value)) => {
                self.write_length_delimited(value.as_bytes(), out)
            }
            (Kind::Bytes, Value::Bytes(value)) => self.write_length_delimited(value, out),
            (kind, value) => unreachable!("{value:?} is not a {kind:?} value"),
        }
    }

    /// Writes a field key.
    fn write_key(&mut self, number: u32, wire_type: WireType, out: &mut Vec<u8>) {
        self.write_varint(u64::from((number << 3) | wire_type as u32), out);
    }

    /// Writes a length-delimited value.
    fn write_length_delimited(&mut self, value: &[u8], out: &mut Vec<u8>) {
        self.write_varint(value.len() as u64, out);
        out.extend_from_slice(value);
    }

    /// Writes a varint, sometimes padded with redundant zero groups up to the limit of ten bytes,
    /// which parsers accept.
    fn write_varint(&mut self, value: u64, out: &mut Vec<u8>) {
        let start = out.len();
        encode_varint(value, out);
        let len = out.len() - start;
        if len < 10 && self.chance(0.1) {
            let padding = 1 + self.below(10 - len);
            if let Some(last) = out.last_mut() {
                *last |= 0x80;
            }
            out.extend(repeat_n(0x80, padding - 1));
            out.push(0);
        }
    }
}

/// Whether `field` holds at most one scalar value.
fn is_singular_scalar(field: &FieldDescriptor) -> bool {
    !field.is_list() && !field.is_map() && field.kind().as_message().is_none()
}

/// Whether repeated values of `kind` can be packed.
fn is_packable(kind: &Kind) -> bool {
    !matches!(kind, Kind::String | Kind::Bytes | Kind::Message(_))
}

/// Whether `value` is the `default` that a map entry may leave out, which a `-0.0` is not,
/// although it compares equal.
fn is_default(value: &Value, default: &Value) -> bool {
    let negative_zero = match value {
        Value::F64(value) => value.to_bits() == (-0.0f64).to_bits(),
        Value::F32(value) => value.to_bits() == (-0.0f32).to_bits(),
        _ => false,
    };
    value == default && !negative_zero
}

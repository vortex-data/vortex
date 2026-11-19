// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Protobuf wire format primitives that the decoder and the encoder share.

use prost::encoding::DecodeContext;
use prost::encoding::WireType;
use prost::encoding::decode_key;
use prost::encoding::decode_varint;
use prost::encoding::skip_field;
use prost_reflect::EnumDescriptor;
use prost_reflect::Kind;
use prost_reflect::Syntax;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// The kind of a protobuf scalar value, which is any field type but a message or a group.
///
/// The kind fixes the wire type of a value and how it maps to its storage value. An enum value is
/// its `int32` number on the wire and in storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScalarKind {
    /// `double`, a 64-bit float.
    Double,
    /// `float`, a 32-bit float.
    Float,
    /// `int32`, a varint that keeps 32 bits.
    Int32,
    /// `int64`, a varint.
    Int64,
    /// `uint32`, a varint that keeps 32 bits.
    Uint32,
    /// `uint64`, a varint.
    Uint64,
    /// `sint32`, a zigzag varint that keeps 32 bits.
    Sint32,
    /// `sint64`, a zigzag varint.
    Sint64,
    /// `fixed32`, four little-endian bytes.
    Fixed32,
    /// `fixed64`, eight little-endian bytes.
    Fixed64,
    /// `sfixed32`, four little-endian bytes.
    Sfixed32,
    /// `sfixed64`, eight little-endian bytes.
    Sfixed64,
    /// `bool`, a varint that is true unless zero.
    Bool,
    /// An enum number, encoded as an `int32`.
    Enum,
    /// `string`, length-delimited UTF-8.
    String,
    /// `bytes`, length-delimited.
    Bytes,
}

impl ScalarKind {
    /// Returns the scalar kind of `kind`, or `None` if `kind` is a message.
    pub(crate) fn of(kind: &Kind) -> Option<Self> {
        Some(match kind {
            Kind::Double => Self::Double,
            Kind::Float => Self::Float,
            Kind::Int32 => Self::Int32,
            Kind::Int64 => Self::Int64,
            Kind::Uint32 => Self::Uint32,
            Kind::Uint64 => Self::Uint64,
            Kind::Sint32 => Self::Sint32,
            Kind::Sint64 => Self::Sint64,
            Kind::Fixed32 => Self::Fixed32,
            Kind::Fixed64 => Self::Fixed64,
            Kind::Sfixed32 => Self::Sfixed32,
            Kind::Sfixed64 => Self::Sfixed64,
            Kind::Bool => Self::Bool,
            Kind::Enum(_) => Self::Enum,
            Kind::String => Self::String,
            Kind::Bytes => Self::Bytes,
            Kind::Message(_) => return None,
        })
    }

    /// The wire type of a single value.
    pub(crate) fn wire_type(self) -> WireType {
        match self {
            Self::Double | Self::Fixed64 | Self::Sfixed64 => WireType::SixtyFourBit,
            Self::Float | Self::Fixed32 | Self::Sfixed32 => WireType::ThirtyTwoBit,
            Self::Int32
            | Self::Int64
            | Self::Uint32
            | Self::Uint64
            | Self::Sint32
            | Self::Sint64
            | Self::Bool
            | Self::Enum => WireType::Varint,
            Self::String | Self::Bytes => WireType::LengthDelimited,
        }
    }

    /// Whether repeated values of this kind can be packed into one length-delimited run.
    pub(crate) fn is_packable(self) -> bool {
        !matches!(self, Self::String | Self::Bytes)
    }
}

/// The numbers that a closed enum defines.
///
/// A proto2 enum is closed: a parser treats a number that the enum does not define like an
/// unknown field, so the number never becomes the value of a field.
#[derive(Clone, Debug)]
pub(crate) struct ClosedEnum {
    /// The defined numbers, sorted and without duplicates.
    numbers: Box<[i32]>,
}

impl ClosedEnum {
    /// Returns the defined numbers of `enum_type`, or `None` if the enum is open.
    pub(crate) fn of(enum_type: &EnumDescriptor) -> Option<Self> {
        if enum_type.parent_file().syntax() != Syntax::Proto2 {
            return None;
        }
        let mut numbers = enum_type
            .values()
            .map(|value| value.number())
            .collect::<Vec<_>>();
        numbers.sort_unstable();
        numbers.dedup();
        Some(Self {
            numbers: numbers.into_boxed_slice(),
        })
    }

    /// Whether the enum defines `number`.
    pub(crate) fn contains(&self, number: i32) -> bool {
        self.numbers.binary_search(&number).is_ok()
    }
}

/// Reads a varint.
pub(crate) fn read_varint(input: &mut &[u8]) -> VortexResult<u64> {
    Ok(decode_varint(input)?)
}

/// Reads four little-endian bytes.
pub(crate) fn read_fixed32(input: &mut &[u8]) -> VortexResult<u32> {
    let Some((bytes, rest)) = input.split_first_chunk() else {
        vortex_bail!("Truncated protobuf input: a 32-bit value needs 4 bytes");
    };
    *input = rest;
    Ok(u32::from_le_bytes(*bytes))
}

/// Reads eight little-endian bytes.
pub(crate) fn read_fixed64(input: &mut &[u8]) -> VortexResult<u64> {
    let Some((bytes, rest)) = input.split_first_chunk() else {
        vortex_bail!("Truncated protobuf input: a 64-bit value needs 8 bytes");
    };
    *input = rest;
    Ok(u64::from_le_bytes(*bytes))
}

/// Reads a length-delimited value: a varint length and that many bytes.
pub(crate) fn read_length_delimited<'a>(input: &mut &'a [u8]) -> VortexResult<&'a [u8]> {
    let len = read_varint(input)?;
    let Some(len) = usize::try_from(len).ok().filter(|&len| len <= input.len()) else {
        vortex_bail!(
            "Truncated protobuf input: a length-delimited value of {len} bytes exceeds the {} \
             remaining bytes",
            input.len()
        );
    };
    let (value, rest) = input.split_at(len);
    *input = rest;
    Ok(value)
}

/// Reads the fields of a group of field `number` up to its end group tag, and returns them
/// without the tag.
pub(crate) fn read_group<'a>(input: &mut &'a [u8], number: u32) -> VortexResult<&'a [u8]> {
    let start = *input;
    while !input.is_empty() {
        let fields = start.len() - input.len();
        let (inner, wire_type) = decode_key(input)?;
        if wire_type == WireType::EndGroup {
            if inner == number {
                return Ok(&start[..fields]);
            }
            vortex_bail!("Unexpected protobuf end group tag of field {inner}");
        }
        skip_field(wire_type, inner, input, DecodeContext::default())?;
    }
    vortex_bail!("Truncated protobuf input: group field {number} has no end group tag")
}

/// Returns the bytes of `start` that were read when `rest` remains.
pub(crate) fn consumed<'a>(start: &'a [u8], rest: &[u8]) -> &'a [u8] {
    &start[..start.len() - rest.len()]
}

/// Keeps the low 32 bits of a varint, as protobuf does for 32-bit integer and enum fields.
#[expect(
    clippy::cast_possible_truncation,
    reason = "protobuf truncates 32-bit varints"
)]
pub(crate) fn varint_to_i32(value: u64) -> i32 {
    value as i32
}

/// Keeps the low 32 bits of a varint, as protobuf does for `uint32` fields.
#[expect(
    clippy::cast_possible_truncation,
    reason = "protobuf truncates 32-bit varints"
)]
pub(crate) fn varint_to_u32(value: u64) -> u32 {
    value as u32
}

/// Decodes a zigzag-encoded `sint32` from the low 32 bits of a varint.
pub(crate) fn zigzag_decode32(value: u64) -> i32 {
    let value = varint_to_u32(value);
    (value >> 1).cast_signed() ^ -(value & 1).cast_signed()
}

/// Decodes a zigzag-encoded `sint64`.
pub(crate) fn zigzag_decode64(value: u64) -> i64 {
    (value >> 1).cast_signed() ^ -(value & 1).cast_signed()
}

/// Zigzag-encodes an `sint32`.
pub(crate) fn zigzag_encode32(value: i32) -> u64 {
    u64::from(((value << 1) ^ (value >> 31)).cast_unsigned())
}

/// Zigzag-encodes an `sint64`.
pub(crate) fn zigzag_encode64(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)).cast_unsigned()
}

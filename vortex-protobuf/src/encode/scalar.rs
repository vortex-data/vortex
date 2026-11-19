// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encoding single protobuf scalar values and field keys.

use prost::encoding::WireType;
use prost::encoding::encode_key;
use prost::encoding::encode_varint;
use prost::encoding::encoded_len_varint;
use prost::encoding::key_len;
use vortex_error::vortex_panic;

use crate::wire::ScalarKind;
use crate::wire::zigzag_encode32;
use crate::wire::zigzag_encode64;

/// A scalar value read from its storage, borrowed for strings and bytes.
#[derive(Clone, Copy, Debug)]
pub(super) enum ScalarRef<'a> {
    /// A `double` value.
    F64(f64),
    /// A `float` value.
    F32(f32),
    /// An `int32`, `sint32`, `sfixed32` or enum value.
    I32(i32),
    /// An `int64`, `sint64` or `sfixed64` value.
    I64(i64),
    /// A `uint32` or `fixed32` value.
    U32(u32),
    /// A `uint64` or `fixed64` value.
    U64(u64),
    /// A `bool` value.
    Bool(bool),
    /// A `string` or `bytes` value.
    Bytes(&'a [u8]),
}

impl ScalarRef<'_> {
    /// Whether a field with implicit presence omits the value: whether it is zero, false or
    /// empty.
    ///
    /// A `-0.0` is not omitted, so that it survives a round trip. Neither is a NaN.
    pub(super) fn is_default(self) -> bool {
        match self {
            Self::F64(value) => value.to_bits() == 0,
            Self::F32(value) => value.to_bits() == 0,
            Self::I32(value) => value == 0,
            Self::I64(value) => value == 0,
            Self::U32(value) => value == 0,
            Self::U64(value) => value == 0,
            Self::Bool(value) => !value,
            Self::Bytes(value) => value.is_empty(),
        }
    }

    /// The number of bytes of the value as `kind`, without a field key.
    pub(super) fn encoded_len(self, kind: ScalarKind) -> usize {
        match kind.wire_type() {
            WireType::SixtyFourBit => 8,
            WireType::ThirtyTwoBit => 4,
            WireType::LengthDelimited => match self {
                Self::Bytes(value) => encoded_len_varint(value.len() as u64) + value.len(),
                _ => mismatch(kind, self),
            },
            _ => encoded_len_varint(self.varint(kind)),
        }
    }

    /// Writes the value as `kind`, without a field key.
    pub(super) fn write(self, kind: ScalarKind, buf: &mut Vec<u8>) {
        match (kind, self) {
            (ScalarKind::Double, Self::F64(value)) => buf.extend_from_slice(&value.to_le_bytes()),
            (ScalarKind::Float, Self::F32(value)) => buf.extend_from_slice(&value.to_le_bytes()),
            (ScalarKind::Fixed32, Self::U32(value)) => buf.extend_from_slice(&value.to_le_bytes()),
            (ScalarKind::Fixed64, Self::U64(value)) => buf.extend_from_slice(&value.to_le_bytes()),
            (ScalarKind::Sfixed32, Self::I32(value)) => buf.extend_from_slice(&value.to_le_bytes()),
            (ScalarKind::Sfixed64, Self::I64(value)) => buf.extend_from_slice(&value.to_le_bytes()),
            (ScalarKind::String | ScalarKind::Bytes, Self::Bytes(value)) => {
                encode_varint(value.len() as u64, buf);
                buf.extend_from_slice(value);
            }
            _ => encode_varint(self.varint(kind), buf),
        }
    }

    /// The varint that encodes the value as the varint kind `kind`.
    fn varint(self, kind: ScalarKind) -> u64 {
        match (kind, self) {
            // Negative `int32` values are sign-extended to ten bytes, as protobuf requires.
            (ScalarKind::Int32 | ScalarKind::Enum, Self::I32(value)) => {
                i64::from(value).cast_unsigned()
            }
            (ScalarKind::Sint32, Self::I32(value)) => zigzag_encode32(value),
            (ScalarKind::Int64, Self::I64(value)) => value.cast_unsigned(),
            (ScalarKind::Sint64, Self::I64(value)) => zigzag_encode64(value),
            (ScalarKind::Uint32, Self::U32(value)) => u64::from(value),
            (ScalarKind::Uint64, Self::U64(value)) => value,
            (ScalarKind::Bool, Self::Bool(value)) => u64::from(value),
            _ => mismatch(kind, self),
        }
    }
}

/// Panics on a storage value that does not match its protobuf kind, which the storage dtype rules
/// out.
fn mismatch(kind: ScalarKind, value: ScalarRef<'_>) -> ! {
    vortex_panic!("Protobuf {kind:?} values cannot be stored as {value:?}")
}

/// Writes the fields of a wrapper message of `kind` that holds `value`.
///
/// The `value` field has implicit presence, so a default value leaves the message empty.
pub(super) fn write_wrapper(kind: ScalarKind, value: ScalarRef<'_>, buf: &mut Vec<u8>) {
    if !value.is_default() {
        write_key(1, kind.wire_type(), buf);
        value.write(kind, buf);
    }
}

/// The number of bytes of a field key with the field number `number`.
pub(super) fn field_key_len(number: u32) -> usize {
    key_len(number)
}

/// Writes a field key.
pub(super) fn write_key(number: u32, wire_type: WireType, buf: &mut Vec<u8>) {
    encode_key(number, wire_type, buf);
}

/// The number of bytes of a length-delimited value of `len` bytes, with its length but without a
/// field key.
pub(super) fn length_delimited_len(len: usize) -> usize {
    encoded_len_varint(len as u64) + len
}

/// Writes the length of a length-delimited value.
pub(super) fn write_len(len: usize, buf: &mut Vec<u8>) {
    encode_varint(len as u64, buf);
}

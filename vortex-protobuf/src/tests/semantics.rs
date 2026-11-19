// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Protobuf parse semantics on hand-written encodings.

use prost::encoding::WireType;
use prost::encoding::encode_varint;
use prost_reflect::DynamicMessage;
use rstest::rstest;
use vortex_array::VortexSessionExecute;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Nullability;
use vortex_array::expr::root;
use vortex_error::VortexResult;

use super::SESSION;
use super::differential::ALL_TYPES;
use super::differential::CLOSED;
use super::differential::LEGACY;
use super::differential::Sample;
use super::differential::Schema;
use super::len_field;
use super::varint_field;
use super::wire_field;
use crate::protobuf_field;

/// Encodes a `test.Inner` message.
fn inner(name: &str, count: u64) -> Vec<u8> {
    [len_field(1, name.as_bytes()), varint_field(2, count)].concat()
}

/// Encodes a map entry with the given key and value fields.
fn entry(fields: &[Vec<u8>]) -> Vec<u8> {
    fields.concat()
}

/// Encodes the payload of a packed run of varints.
fn packed(values: &[u64]) -> Vec<u8> {
    let mut payload = Vec::new();
    for &value in values {
        encode_varint(value, &mut payload);
    }
    payload
}

/// The decoder reads `input` as one message whose field at `path` displays as `expected`, as the
/// `prost-reflect` oracle does too.
#[rstest]
#[case::scalar_last_value_wins(
    ALL_TYPES,
    [varint_field(3, 5), varint_field(3, 7)].concat(),
    "f_int32",
    "7i32"
)]
#[case::string_last_value_wins(
    ALL_TYPES,
    [len_field(14, b"a"), len_field(14, b"bc")].concat(),
    "f_string",
    "\"bc\""
)]
#[case::default_after_value(
    ALL_TYPES,
    [varint_field(4, 9), varint_field(4, 0)].concat(),
    "f_int64",
    "0i64"
)]
#[case::optional_default_after_value(
    ALL_TYPES,
    [varint_field(26, 9), varint_field(26, 0)].concat(),
    "o_int32",
    "0i32"
)]
#[case::message_occurrences_merge(
    ALL_TYPES,
    [len_field(17, &len_field(1, b"a")), len_field(17, &varint_field(2, 2))].concat(),
    "inner",
    "{name: \"a\", count: 2i64}"
)]
#[case::nested_lists_append(
    ALL_TYPES,
    [
        len_field(18, &len_field(2, &inner("x", 1))),
        varint_field(3, 1),
        len_field(18, &len_field(2, &inner("y", 2))),
    ]
    .concat(),
    "outer.items",
    "[{name: \"x\", count: 1i64}, {name: \"y\", count: 2i64}]"
)]
#[case::oneof_last_member_wins(
    ALL_TYPES,
    [len_field(25, &inner("a", 1)), len_field(24, b"t"), len_field(25, &len_field(1, b"b"))].concat(),
    "c_inner",
    "{name: \"b\", count: 0i64}"
)]
#[case::oneof_member_is_cleared(
    ALL_TYPES,
    [len_field(24, b"t"), len_field(25, &inner("a", 1))].concat(),
    "c_text",
    "null"
)]
#[case::oneof_message_member_merges(
    ALL_TYPES,
    [len_field(25, &len_field(1, b"a")), len_field(25, &varint_field(2, 3))].concat(),
    "c_inner",
    "{name: \"a\", count: 3i64}"
)]
#[case::packed_and_unpacked_values_mix(
    ALL_TYPES,
    [varint_field(19, 1), len_field(19, &packed(&[2, 3])), len_field(19, &[]), varint_field(19, 4)]
        .concat(),
    "r_int32",
    "[1i32, 2i32, 3i32, 4i32]"
)]
#[case::map_entries(
    ALL_TYPES,
    [
        len_field(23, &entry(&[varint_field(1, 2), len_field(2, b"b")])),
        len_field(23, &entry(&[len_field(2, b"a"), varint_field(1, 1)])),
        len_field(23, &entry(&[len_field(2, b"d")])),
        len_field(23, &entry(&[varint_field(1, 3), varint_field(9, 1)])),
        varint_field(3, 1),
        len_field(23, &entry(&[varint_field(1, 1), len_field(2, b"c")])),
    ]
    .concat(),
    "m_labels",
    "[{key: 0i32, value: \"d\"}, {key: 1i32, value: \"c\"}, {key: 2i32, value: \"b\"}, \
     {key: 3i32, value: \"\"}]"
)]
#[case::map_message_values_default(
    ALL_TYPES,
    len_field(22, &entry(&[len_field(1, b"k")])),
    "m_inner",
    "[{key: \"k\", value: {name: \"\", count: 0i64}}]"
)]
#[case::unknown_fields_are_skipped(
    ALL_TYPES,
    [
        varint_field(99, 1),
        wire_field(98, WireType::StartGroup, &varint_field(1, 1)),
        wire_field(98, WireType::EndGroup, &[]),
        len_field(97, b"xyz"),
        wire_field(96, WireType::SixtyFourBit, &[0; 8]),
        varint_field(3, 4),
    ]
    .concat(),
    "f_int32",
    "4i32"
)]
fn test_parse_semantics(
    #[case] schema: Schema,
    #[case] input: Vec<u8>,
    #[case] path: &str,
    #[case] expected: &str,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let sample = Sample::new(schema, Nullability::NonNullable)?;
    let rows = [Some(input)];
    let array = sample.decode(&rows)?;
    assert_arrays_eq!(array, sample.oracle(&rows)?, &mut ctx);
    let value = array
        .apply(&protobuf_field(path, root()))?
        .execute_scalar(0, &mut ctx)?;
    assert_eq!(value.to_string(), expected);
    Ok(())
}

/// A value whose wire type does not match its field is skipped like an unknown field, as
/// protobuf parsers do, while `prost-reflect` rejects the message. So is a number that a closed
/// enum does not define, which `prost-reflect` keeps: it leaves a singular field as it was, drops
/// a list element and drops a map entry.
#[rstest]
#[case::mismatched_scalar(
    ALL_TYPES,
    [varint_field(3, 5), len_field(3, b"ab"), wire_field(4, WireType::ThirtyTwoBit, &[1; 4])]
        .concat(),
    "f_int32",
    "5i32"
)]
#[case::mismatched_message(ALL_TYPES, varint_field(17, 1), "inner", "null")]
#[case::mismatched_group(
    LEGACY,
    len_field(5, &varint_field(6, 1)),
    "settings",
    "null"
)]
#[case::mismatched_list_element(
    ALL_TYPES,
    [varint_field(19, 1), wire_field(19, WireType::SixtyFourBit, &[1; 8])].concat(),
    "r_int32",
    "[1i32]"
)]
#[case::mismatched_map_value(
    ALL_TYPES,
    len_field(23, &entry(&[varint_field(1, 1), varint_field(2, 7)])),
    "m_labels",
    "[{key: 1i32, value: \"\"}]"
)]
#[case::closed_enum_keeps_value(
    CLOSED,
    [varint_field(1, 2), varint_field(1, 3)].concat(),
    "mode",
    "2i32"
)]
#[case::closed_enum_stays_absent(CLOSED, varint_field(1, 7), "mode", "null")]
#[case::closed_enum_list(
    CLOSED,
    [varint_field(2, 1), varint_field(2, 3), varint_field(2, 5)].concat(),
    "modes",
    "[1i32, 5i32]"
)]
#[case::closed_enum_packed_list(
    CLOSED,
    len_field(3, &packed(&[1, 4, 2, u64::MAX])),
    "packed_modes",
    "[1i32, 2i32]"
)]
#[case::closed_enum_map(
    CLOSED,
    [
        len_field(4, &entry(&[len_field(1, b"a"), varint_field(2, 9)])),
        len_field(4, &entry(&[len_field(1, b"b"), varint_field(2, 2)])),
        len_field(4, &entry(&[len_field(1, b"c")])),
        len_field(4, &entry(&[len_field(1, b"d"), varint_field(2, 2), varint_field(2, 4)])),
    ]
    .concat(),
    "named",
    "[{key: \"b\", value: 2i32}, {key: \"c\", value: 1i32}]"
)]
#[case::closed_enum_oneof(
    CLOSED,
    [len_field(10, b"x"), varint_field(9, 9)].concat(),
    "choice_text",
    "\"x\""
)]
fn test_parse_semantics_beyond_prost_reflect(
    #[case] schema: Schema,
    #[case] input: Vec<u8>,
    #[case] path: &str,
    #[case] expected: &str,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let sample = Sample::new(schema, Nullability::NonNullable)?;
    let array = sample.decode(&[Some(input.clone())])?;
    let value = array
        .apply(&protobuf_field(path, root()))?
        .execute_scalar(0, &mut ctx)?;
    assert_eq!(value.to_string(), expected);

    let parsed = DynamicMessage::decode(sample.descriptor().clone(), input.as_slice());
    if schema == CLOSED {
        assert!(parsed.is_ok());
    } else {
        assert!(parsed.is_err());
    }
    Ok(())
}

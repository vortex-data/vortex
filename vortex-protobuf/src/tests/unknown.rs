// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unknown fields, which the `preserve_unknown_fields` option stores and writes back.

use insta::assert_snapshot;
use prost::Message;
use prost::encoding::WireType;
use prost_reflect::DynamicMessage;
use prost_reflect::MessageDescriptor;
use rstest::rstest;
use vortex_array::VortexSessionExecute;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Nullability;
use vortex_array::expr::root;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::SESSION;
use super::decode_rows;
use super::describe_fields;
use super::differential::ALL_TYPES;
use super::differential::CLOSED;
use super::differential::LEGACY;
use super::differential::REPEATED;
use super::differential::Sample;
use super::differential::Schema;
use super::encode_rows;
use super::generate::Generator;
use super::len_field;
use super::message;
use super::preserving;
use super::protobuf_dtype_with;
use super::varint_field;
use super::wire_field;
use crate::ProtobufDecoder;
use crate::protobuf_field;

/// A schema that adds fields to [`OLDER_PROTO`]: scalar, repeated and map fields at the top
/// level, and fields of the nested message, which occurs singular, repeated, as a map value and
/// as a oneof member.
const NEWER_PROTO: (&str, &str) = (
    "evolved.proto",
    r#"
syntax = "proto3";
package evolved;

enum Kind {
  KIND_UNSPECIFIED = 0;
  KIND_A = 1;
}

message Detail {
  string note = 1;
  int64 extra = 2;
  repeated string tags = 3;
  Kind kind = 4;
  Detail child = 5;
}

message Event {
  int64 id = 1;
  Detail detail = 2;
  repeated Detail details = 3;
  map<string, Detail> by_name = 4;
  oneof choice {
    string text = 5;
    Detail nested = 6;
  }
  string label = 7;
  repeated sint32 scores = 8;
  Detail added = 9;
  fixed64 checksum = 10;
  float ratio = 11;
  map<int32, string> notes = 12;
  optional bool flag = 13;
}
"#,
);

/// The schema that [`NEWER_PROTO`] extends.
const OLDER_PROTO: (&str, &str) = (
    "evolved.proto",
    r#"
syntax = "proto3";
package evolved;

message Detail {
  string note = 1;
}

message Event {
  int64 id = 1;
  Detail detail = 2;
  repeated Detail details = 3;
  map<string, Detail> by_name = 4;
  oneof choice {
    string text = 5;
    Detail nested = 6;
  }
}
"#,
);

/// Parses `bytes` as a message of type `descriptor` with `prost-reflect`, in a normal form.
///
/// `prost-reflect` keeps an empty packed run as a present empty list, which equals an absent
/// field in protobuf, so the message is encoded and parsed once more, which drops such lists.
fn parse(descriptor: &MessageDescriptor, bytes: &[u8]) -> VortexResult<DynamicMessage> {
    let message = DynamicMessage::decode(descriptor.clone(), bytes)?;
    Ok(DynamicMessage::decode(
        descriptor.clone(),
        message.encode_to_vec().as_slice(),
    )?)
}

/// Checks that `actual` parses to the same message as `expected`, without printing the messages,
/// whose descriptors may be recursive.
fn assert_parses_same(
    descriptor: &MessageDescriptor,
    actual: &[u8],
    expected: &[u8],
    row: usize,
) -> VortexResult<()> {
    assert!(
        parse(descriptor, actual)? == parse(descriptor, expected)?,
        "row {row}: {actual:?} does not parse as {expected:?}"
    );
    Ok(())
}

/// Every message gets a `$unknown` binary field after its fields, but map entries do not.
#[test]
fn test_unknown_fields_storage_dtype() -> VortexResult<()> {
    let dtype = protobuf_dtype_with(
        &[OLDER_PROTO],
        "evolved.Event",
        Nullability::NonNullable,
        preserving(),
    )?;
    assert_snapshot!(describe_fields(dtype.storage_dtype()), @r"
    id: i64
    detail: {note=utf8, $unknown=binary}?
    details: list({note=utf8, $unknown=binary})
    by_name: list({key=utf8, value={note=utf8, $unknown=binary}})
    text: utf8?
    nested: {note=utf8, $unknown=binary}?
    $unknown: binary
    ");
    Ok(())
}

/// Messages of a newer schema, decoded and encoded again with an older schema, parse to the
/// same messages with the newer schema: the fields that the older schema does not know survive,
/// at every message level.
#[rstest]
fn test_older_schema_keeps_newer_fields(
    #[values(false, true)] non_canonical: bool,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let newer = message(&[NEWER_PROTO], "evolved.Event")?;
    let older = protobuf_dtype_with(&[OLDER_PROTO], "evolved.Event", nullability, preserving())?;
    let mut generator = Generator::new(11);
    let rows = (0..200)
        .map(|_| {
            (!nullability.is_nullable() || generator.chance(0.8)).then(|| {
                let message = generator.message(&newer);
                if non_canonical {
                    generator.non_canonical(&message)
                } else {
                    message.encode_to_vec()
                }
            })
        })
        .collect::<Vec<_>>();

    let decoded = decode_rows(&older, &rows)?;
    let encoded = encode_rows(&decoded)?;
    for (row, (input, encoded)) in rows.iter().zip(&encoded).enumerate() {
        match (input, encoded) {
            (Some(input), Some(encoded)) => assert_parses_same(&newer, encoded, input, row)?,
            (input, encoded) => assert_eq!(input.is_none(), encoded.is_none(), "row {row}"),
        }
    }
    assert_arrays_eq!(decode_rows(&older, &encoded)?, decoded, &mut ctx);
    Ok(())
}

/// With unknown fields stored, the messages of every test schema survive a round trip in their
/// non-canonical encodings, which hold unknown fields and groups at every level.
#[rstest]
fn test_round_trip_keeps_unknown_fields(
    #[values(ALL_TYPES, REPEATED, LEGACY, CLOSED)] schema: Schema,
) -> VortexResult<()> {
    let (sources, name) = schema;
    let mut sample = Sample::new(schema, Nullability::NonNullable)?;
    let dtype = protobuf_dtype_with(sources, name, Nullability::NonNullable, preserving())?;
    let rows = sample.non_canonical();
    let encoded = encode_rows(&decode_rows(&dtype, &rows)?)?;
    let descriptor = sample.descriptor();
    for (row, (input, encoded)) in rows.iter().zip(&encoded).enumerate() {
        let (Some(input), Some(encoded)) = (input, encoded) else {
            return Err(vortex_err!("row {row} is null"));
        };
        assert_parses_same(descriptor, encoded, input, row)?;
    }
    Ok(())
}

/// Encodes a map entry with the given fields.
fn entry(fields: &[Vec<u8>]) -> Vec<u8> {
    fields.concat()
}

/// The unknown fields of a message keep their bytes, in the order they were read, at the message
/// level where they occur. So do values whose wire type does not match their field, and numbers
/// that a closed enum does not define, as protobuf parsers keep them: a single number as it was
/// read, the numbers of a packed run as separate varint fields, and a map entry as a whole.
#[rstest]
#[case::each_wire_type(
    ALL_TYPES,
    [
        varint_field(99, 1),
        varint_field(3, 4),
        wire_field(98, WireType::StartGroup, &varint_field(1, 1)),
        wire_field(98, WireType::EndGroup, &[]),
        len_field(97, b"xyz"),
        wire_field(96, WireType::SixtyFourBit, &[7; 8]),
        wire_field(95, WireType::ThirtyTwoBit, &[8; 4]),
    ]
    .concat(),
    "$unknown",
    [
        varint_field(99, 1),
        wire_field(98, WireType::StartGroup, &varint_field(1, 1)),
        wire_field(98, WireType::EndGroup, &[]),
        len_field(97, b"xyz"),
        wire_field(96, WireType::SixtyFourBit, &[7; 8]),
        wire_field(95, WireType::ThirtyTwoBit, &[8; 4]),
    ]
    .concat()
)]
#[case::mismatched_wire_type(
    ALL_TYPES,
    [len_field(3, b"ab"), varint_field(3, 5)].concat(),
    "$unknown",
    len_field(3, b"ab")
)]
#[case::nested_message(
    ALL_TYPES,
    [
        len_field(17, &[len_field(1, b"a"), varint_field(9, 5)].concat()),
        len_field(17, &varint_field(10, 6)),
    ]
    .concat(),
    "inner.$unknown",
    [varint_field(9, 5), varint_field(10, 6)].concat()
)]
#[case::group(
    LEGACY,
    [
        wire_field(5, WireType::StartGroup, &varint_field(9, 1)),
        wire_field(5, WireType::EndGroup, &[]),
    ]
    .concat(),
    "settings.$unknown",
    varint_field(9, 1)
)]
#[case::closed_enum(
    CLOSED,
    [varint_field(1, 7), varint_field(1, 2)].concat(),
    "$unknown",
    varint_field(1, 7)
)]
#[case::closed_enum_packed(
    CLOSED,
    len_field(3, &[1, 4, 2, 6]),
    "$unknown",
    [varint_field(3, 4), varint_field(3, 6)].concat()
)]
#[case::closed_enum_map_entry(
    CLOSED,
    [
        len_field(4, &entry(&[len_field(1, b"a"), varint_field(2, 9)])),
        len_field(4, &entry(&[len_field(1, b"b"), varint_field(2, 2)])),
    ]
    .concat(),
    "$unknown",
    len_field(4, &entry(&[len_field(1, b"a"), varint_field(2, 9)]))
)]
#[case::map_entry_fields_are_dropped(
    ALL_TYPES,
    len_field(23, &entry(&[varint_field(1, 1), varint_field(9, 1)])),
    "$unknown",
    Vec::new()
)]
fn test_unknown_fields_keep_their_bytes(
    #[case] schema: Schema,
    #[case] input: Vec<u8>,
    #[case] path: &str,
    #[case] expected: Vec<u8>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (sources, name) = schema;
    let dtype = protobuf_dtype_with(sources, name, Nullability::NonNullable, preserving())?;
    let rows = [Some(input)];
    let array = decode_rows(&dtype, &rows)?;
    let unknown = array
        .clone()
        .apply(&protobuf_field(path, root()))?
        .execute_scalar(0, &mut ctx)?;
    assert_eq!(
        unknown.as_binary().value().map(|value| value.as_slice()),
        Some(expected.as_slice())
    );

    let encoded = encode_rows(&array)?;
    assert_arrays_eq!(decode_rows(&dtype, &encoded)?, array, &mut ctx);
    Ok(())
}

/// The encoder writes the unknown fields of a message after its known fields.
#[test]
fn test_encode_writes_unknown_fields_last() -> VortexResult<()> {
    let (sources, name) = LEGACY;
    let dtype = protobuf_dtype_with(sources, name, Nullability::NonNullable, preserving())?;
    let input = [varint_field(99, 1), varint_field(1, 2), len_field(97, b"x")].concat();
    let encoded = encode_rows(&decode_rows(&dtype, &[Some(input)])?)?;
    assert_eq!(
        encoded,
        [Some(
            [varint_field(1, 2), varint_field(99, 1), len_field(97, b"x")].concat()
        )]
    );
    Ok(())
}

/// A message that fails to decode leaves no unknown fields behind, and no numbers of a packed run
/// that a closed enum does not define, which would otherwise become unknown fields of the next
/// message.
#[rstest]
#[case::unknown_field(
    LEGACY,
    // An unknown field, then a string that runs past the end of the input.
    [varint_field(99, 1), vec![0x12, 0x05, b'a']].concat(),
    [varint_field(1, 2), varint_field(98, 3)].concat()
)]
#[case::closed_enum_packed(
    CLOSED,
    // An undefined number in a packed run, then a truncated varint.
    len_field(3, &[0x04, 0x80]),
    [varint_field(8, 7), varint_field(98, 3)].concat()
)]
fn test_failed_message_leaves_no_unknown_fields(
    #[case] schema: Schema,
    #[case] invalid: Vec<u8>,
    #[case] valid: Vec<u8>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (sources, name) = schema;
    let dtype = protobuf_dtype_with(sources, name, Nullability::NonNullable, preserving())?;
    let mut decoder = ProtobufDecoder::new(&dtype);
    assert!(decoder.append_message(&invalid).is_err());
    decoder.append_message(&valid)?;

    assert_arrays_eq!(
        decoder.finish()?,
        decode_rows(&dtype, &[Some(valid)])?,
        &mut ctx
    );
    Ok(())
}

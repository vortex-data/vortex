// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Recursive messages, which the `max_recursion_depth` option expands to a depth and stores as
//! encoded bytes below it.

use prost::Message;
use prost::encoding::WireType;
use prost_reflect::DynamicMessage;
use prost_reflect::MapKey;
use prost_reflect::MessageDescriptor;
use prost_reflect::Value;
use rstest::rstest;
use vortex_array::VortexSessionExecute;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Nullability;
use vortex_array::expr::root;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::RECURSIVE_PROTO;
use super::SESSION;
use super::build;
use super::decode_rows;
use super::describe_fields;
use super::encode_rows;
use super::len_field;
use super::message;
use super::protobuf_dtype_with;
use super::varint_field;
use super::wire_field;
use crate::MAX_RECURSION_DEPTH;
use crate::MappingOptions;
use crate::ProtobufMetadata;
use crate::protobuf_field;

/// A message that contains itself through a singular, a repeated and a map field, and a group
/// whose type contains itself through a message field.
const CHAIN_PROTO: (&str, &str) = (
    "chain.proto",
    r#"
syntax = "proto2";
package chain;

message Chain {
  optional int32 value = 1;
  optional Chain next = 2;
  repeated Chain children = 3;
  map<string, Chain> named = 4;
}

message Outer {
  optional group Inner = 1 {
    optional int32 value = 2;
    optional Outer outer = 3;
  }
}
"#,
);

/// The mapping options that expand recursive messages to `depth`.
fn recursion_depth(depth: u32) -> MappingOptions {
    MappingOptions {
        max_recursion_depth: Some(depth),
        ..MappingOptions::default()
    }
}

/// Returns a `chain.Chain` message with `value`, whose `next` field holds `levels` more messages
/// with the values after it, and whose children and named entries hold one message each.
fn chain(
    descriptor: &MessageDescriptor,
    value: i32,
    levels: usize,
) -> VortexResult<DynamicMessage> {
    let leaf = |value: i32| -> VortexResult<Value> {
        Ok(Value::Message(build(
            descriptor,
            "chain.Chain",
            [("value", Value::I32(value))],
        )?))
    };
    let mut fields = vec![
        ("value", Value::I32(value)),
        ("children", Value::List(vec![leaf(-value)?])),
        (
            "named",
            Value::Map(
                [(MapKey::String(value.to_string()), leaf(value * 10)?)]
                    .into_iter()
                    .collect(),
            ),
        ),
    ];
    if levels > 0 {
        fields.push((
            "next",
            Value::Message(chain(descriptor, value + 1, levels - 1)?),
        ));
    }
    build(descriptor, "chain.Chain", fields)
}

/// Returns the message in the `next` field of a `chain.Chain` message, if it is set.
fn next_message(message: &DynamicMessage) -> Option<DynamicMessage> {
    if !message.has_field_by_name("next") {
        return None;
    }
    message.get_field_by_name("next")?.as_message().cloned()
}

/// A message occurs as a struct inside itself up to the depth, and deeper occurrences are binary.
#[rstest]
#[case::depth_0(
    0,
    "value: i32?
next: binary?
children: list(binary)
named: list({key=utf8, value=binary})"
)]
#[case::depth_1(1, "value: i32?
next: {value=i32?, next=binary?, children=list(binary), named=list({key=utf8, value=binary})}?
children: list({value=i32?, next=binary?, children=list(binary), named=list({key=utf8, value=binary})})
named: list({key=utf8, value={value=i32?, next=binary?, children=list(binary), named=list({key=utf8, value=binary})}})")]
fn test_recursive_storage_dtype(#[case] depth: u32, #[case] expected: &str) -> VortexResult<()> {
    let dtype = protobuf_dtype_with(
        &[CHAIN_PROTO],
        "chain.Chain",
        Nullability::NonNullable,
        recursion_depth(depth),
    )?;
    assert_eq!(describe_fields(dtype.storage_dtype()), expected);
    Ok(())
}

/// The depth counts the occurrences of each message type on the path from the root, so mutual
/// recursion and recursion through map values and groups expand too.
#[rstest]
#[case::mutual(&[RECURSIVE_PROTO], "recursive.Ping", "pong: {ping=binary?}?")]
#[case::map_value(
    &[RECURSIVE_PROTO],
    "recursive.Tree",
    "branches: list({key=utf8, value=binary})"
)]
#[case::nested(&[RECURSIVE_PROTO], "recursive.Holder", "ping: {pong={ping=binary?}?}?")]
#[case::group(
    &[CHAIN_PROTO],
    "chain.Outer.Inner",
    "value: i32?\nouter: {inner=binary?}?"
)]
fn test_recursive_storage_dtype_by_type(
    #[case] sources: &[(&str, &str)],
    #[case] name: &str,
    #[case] expected: &str,
) -> VortexResult<()> {
    let dtype = protobuf_dtype_with(sources, name, Nullability::NonNullable, recursion_depth(0))?;
    assert_eq!(describe_fields(dtype.storage_dtype()), expected);
    Ok(())
}

/// Recursive messages decode and encode back to the same bytes at every depth, and the binary
/// value at the cut is the encoding of the message there.
#[rstest]
fn test_recursive_round_trip(
    #[values(0, 1, 3)] depth: u32,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let descriptor = message(&[CHAIN_PROTO], "chain.Chain")?;
    let dtype = protobuf_dtype_with(
        &[CHAIN_PROTO],
        "chain.Chain",
        nullability,
        recursion_depth(depth),
    )?;
    let messages = (0..5)
        .map(|levels| chain(&descriptor, 1, levels))
        .collect::<VortexResult<Vec<_>>>()?;
    let rows = messages
        .iter()
        .map(|message| Some(message.encode_to_vec()))
        .collect::<Vec<_>>();
    let array = decode_rows(&dtype, &rows)?;
    // Without maps of several entries, the bytes equal those of `prost-reflect`.
    assert_eq!(encode_rows(&array)?, rows);

    let cut = vec!["next"; depth as usize + 1].join(".");
    let cut_values = array.clone().apply(&protobuf_field(&cut, root()))?;
    for (row, message) in messages.iter().enumerate() {
        let expected = (0..=depth).try_fold(message.clone(), |message, _| next_message(&message));
        let value = cut_values.execute_scalar(row, &mut ctx)?;
        assert_eq!(
            value
                .as_binary()
                .value()
                .map(|bytes| bytes.as_slice().to_vec()),
            expected.map(|message| message.encode_to_vec()),
            "row {row}"
        );
    }
    assert_arrays_eq!(decode_rows(&dtype, &rows)?, array, &mut ctx);
    Ok(())
}

/// Occurrences of a field at the cut merge as their encodings concatenate, and a group at the
/// cut keeps its fields without the group tags.
#[rstest]
#[case::merged_occurrences(
    "chain.Chain",
    [len_field(2, &varint_field(1, 1)), len_field(2, &varint_field(1, 2))].concat(),
    "next",
    [varint_field(1, 1), varint_field(1, 2)].concat()
)]
#[case::group(
    "chain.Outer.Inner",
    len_field(
        3,
        &[
            wire_field(1, WireType::StartGroup, &varint_field(2, 5)),
            wire_field(1, WireType::EndGroup, &[]),
        ]
        .concat(),
    ),
    "outer.inner",
    varint_field(2, 5)
)]
fn test_recursive_cut_holds_encoding(
    #[case] name: &str,
    #[case] input: Vec<u8>,
    #[case] path: &str,
    #[case] expected: Vec<u8>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = protobuf_dtype_with(
        &[CHAIN_PROTO],
        name,
        Nullability::NonNullable,
        recursion_depth(0),
    )?;
    let rows = [Some(input.clone())];
    let array = decode_rows(&dtype, &rows)?;
    let value = array
        .clone()
        .apply(&protobuf_field(path, root()))?
        .execute_scalar(0, &mut ctx)?;
    assert_eq!(
        value.as_binary().value().map(|bytes| bytes.as_slice()),
        Some(expected.as_slice())
    );

    let encoded = encode_rows(&array)?;
    let descriptor = message(&[CHAIN_PROTO], name)?;
    let parse = |bytes: &[u8]| DynamicMessage::decode(descriptor.clone(), bytes);
    let reencoded = encoded[0]
        .as_deref()
        .ok_or_else(|| vortex_err!("message 0 is null"))?;
    assert_eq!(parse(reencoded)?, parse(&input)?);
    Ok(())
}

/// The depth has a limit, and so has the number of storage fields that an expansion creates.
#[rstest]
#[case::depth_limit(
    "recursive.Node",
    101,
    "max_recursion_depth 101 exceeds the limit of 100"
)]
#[case::field_limit(
    "chain.Chain",
    20,
    "The storage of protobuf message chain.Chain needs more than 100000 fields"
)]
fn test_recursion_is_bounded(
    #[case] name: &str,
    #[case] depth: u32,
    #[case] expected: &str,
) -> VortexResult<()> {
    let descriptor = message(&[RECURSIVE_PROTO, CHAIN_PROTO], name)?;
    let error = ProtobufMetadata::try_new_with_options(&descriptor, recursion_depth(depth))
        .err()
        .ok_or_else(|| vortex_err!("{name} at depth {depth} was accepted"))?;
    assert!(error.to_string().contains(expected), "{error}");
    Ok(())
}

/// A message that recurs through one field expands to the largest depth.
#[test]
fn test_recursion_expands_to_largest_depth() -> VortexResult<()> {
    let descriptor = message(&[RECURSIVE_PROTO], "recursive.Node")?;
    let metadata =
        ProtobufMetadata::try_new_with_options(&descriptor, recursion_depth(MAX_RECURSION_DEPTH))?;
    assert_eq!(metadata.options().max_recursion_depth, Some(100));
    Ok(())
}

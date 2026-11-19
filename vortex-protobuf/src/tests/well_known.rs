// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The dedicated mappings of the `google.protobuf` well-known types.

use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use insta::assert_snapshot;
use prost::Message;
use prost::bytes::Bytes;
use prost::encoding::WireType;
use prost_reflect::DynamicMessage;
use prost_reflect::MapKey;
use prost_reflect::MessageDescriptor;
use prost_reflect::Value;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::expr::Expression;
use vortex_array::expr::ext_storage;
use vortex_array::expr::root;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

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
use crate::JsonMapping;
use crate::MappingOptions;
use crate::Protobuf;
use crate::ProtobufDecoder;
use crate::ProtobufMetadata;
use crate::TimestampMapping;
use crate::WrapperMapping;
use crate::protobuf_field;
use crate::well_known::JsonType;
use crate::well_known::decode_json;
use crate::well_known::decode_timestamp;
use crate::well_known::encode_json;
use crate::well_known::encode_timestamp;

/// A schema with every wrapper type, timestamps, a duration and the JSON types, also repeated,
/// as map values, in a oneof and in maps of repeated messages.
pub(super) const KNOWN_PROTO: (&str, &str) = (
    "known.proto",
    r#"
syntax = "proto3";
package known;

import "google/protobuf/duration.proto";
import "google/protobuf/struct.proto";
import "google/protobuf/timestamp.proto";
import "google/protobuf/wrappers.proto";

message Known {
  google.protobuf.DoubleValue double_value = 1;
  google.protobuf.FloatValue float_value = 2;
  google.protobuf.Int64Value int64_value = 3;
  google.protobuf.UInt64Value uint64_value = 4;
  google.protobuf.Int32Value int32_value = 5;
  google.protobuf.UInt32Value uint32_value = 6;
  google.protobuf.BoolValue bool_value = 7;
  google.protobuf.StringValue string_value = 8;
  google.protobuf.BytesValue bytes_value = 9;
  google.protobuf.Timestamp created = 10;
  google.protobuf.Duration elapsed = 11;
  google.protobuf.Struct attributes = 12;
  google.protobuf.Value payload = 13;
  google.protobuf.ListValue items = 14;
  repeated google.protobuf.Timestamp history = 15;
  map<string, google.protobuf.Int32Value> counts = 16;
  oneof when {
    google.protobuf.Timestamp at = 17;
    string never = 18;
  }
  map<string, google.protobuf.Timestamp> times = 19;
  repeated Bag bags = 20;
}

message Bag {
  map<string, google.protobuf.Value> values = 1;
}
"#,
);

/// The mapping options that store every well-known type in its dedicated form.
pub(super) fn dedicated() -> MappingOptions {
    MappingOptions {
        wrappers: WrapperMapping::Primitive,
        timestamps: TimestampMapping::Nanoseconds,
        json: JsonMapping::Text,
        ..MappingOptions::default()
    }
}

/// Builds a message of the well-known type `name` with the given fields.
fn known(
    descriptor: &MessageDescriptor,
    name: &str,
    fields: impl IntoIterator<Item = (&'static str, Value)>,
) -> VortexResult<Value> {
    build(descriptor, &format!("google.protobuf.{name}"), fields).map(Value::Message)
}

/// Builds a `google.protobuf.Timestamp`.
fn timestamp(descriptor: &MessageDescriptor, seconds: i64, nanos: i32) -> VortexResult<Value> {
    known(
        descriptor,
        "Timestamp",
        [
            ("seconds", Value::I64(seconds)),
            ("nanos", Value::I32(nanos)),
        ],
    )
}

/// Builds a `google.protobuf.Value` with one kind field.
fn json_value(
    descriptor: &MessageDescriptor,
    kind: &'static str,
    value: Value,
) -> VortexResult<Value> {
    known(descriptor, "Value", [(kind, value)])
}

/// Builds a `google.protobuf.Struct` with the given fields.
fn json_struct(
    descriptor: &MessageDescriptor,
    fields: impl IntoIterator<Item = (&'static str, Value)>,
) -> VortexResult<Value> {
    let fields = fields
        .into_iter()
        .map(|(key, value)| (MapKey::String(key.to_owned()), value))
        .collect();
    known(descriptor, "Struct", [("fields", Value::Map(fields))])
}

/// Serialized `known.Known` messages: one without fields, one with values that are not defaults,
/// and one whose well-known messages are present but hold defaults.
pub(super) fn known_messages(descriptor: &MessageDescriptor) -> VortexResult<Vec<Vec<u8>>> {
    let wrapper = |name: &str, value: Value| known(descriptor, name, [("value", value)]);
    let messages = [
        build(descriptor, "known.Known", [])?,
        build(
            descriptor,
            "known.Known",
            [
                ("double_value", wrapper("DoubleValue", Value::F64(1.5))?),
                ("float_value", wrapper("FloatValue", Value::F32(-2.5))?),
                ("int64_value", wrapper("Int64Value", Value::I64(-3))?),
                (
                    "uint64_value",
                    wrapper("UInt64Value", Value::U64(u64::MAX))?,
                ),
                ("int32_value", wrapper("Int32Value", Value::I32(-4))?),
                ("uint32_value", wrapper("UInt32Value", Value::U32(5))?),
                ("bool_value", wrapper("BoolValue", Value::Bool(true))?),
                (
                    "string_value",
                    wrapper("StringValue", Value::String("s".into()))?,
                ),
                (
                    "bytes_value",
                    wrapper("BytesValue", Value::Bytes(Bytes::from_static(&[1, 2])))?,
                ),
                ("created", timestamp(descriptor, 1_700_000_000, 123)?),
                (
                    "elapsed",
                    known(
                        descriptor,
                        "Duration",
                        [("seconds", Value::I64(5)), ("nanos", Value::I32(6))],
                    )?,
                ),
                (
                    "attributes",
                    json_struct(
                        descriptor,
                        [
                            (
                                "b",
                                json_value(
                                    descriptor,
                                    "string_value",
                                    Value::String("x\"".into()),
                                )?,
                            ),
                            (
                                "a",
                                json_value(descriptor, "number_value", Value::F64(1.0))?,
                            ),
                            (
                                "c",
                                json_value(
                                    descriptor,
                                    "list_value",
                                    known(
                                        descriptor,
                                        "ListValue",
                                        [(
                                            "values",
                                            Value::List(vec![
                                                json_value(
                                                    descriptor,
                                                    "bool_value",
                                                    Value::Bool(true),
                                                )?,
                                                json_value(
                                                    descriptor,
                                                    "null_value",
                                                    Value::EnumNumber(0),
                                                )?,
                                            ]),
                                        )],
                                    )?,
                                )?,
                            ),
                            (
                                "d",
                                json_value(
                                    descriptor,
                                    "struct_value",
                                    json_struct(descriptor, [])?,
                                )?,
                            ),
                        ],
                    )?,
                ),
                (
                    "payload",
                    json_value(descriptor, "number_value", Value::F64(-0.5))?,
                ),
                (
                    "items",
                    known(
                        descriptor,
                        "ListValue",
                        [(
                            "values",
                            Value::List(vec![json_value(
                                descriptor,
                                "string_value",
                                Value::String("é".into()),
                            )?]),
                        )],
                    )?,
                ),
                (
                    "history",
                    Value::List(vec![
                        timestamp(descriptor, 0, 0)?,
                        timestamp(descriptor, -1, 500_000_000)?,
                    ]),
                ),
                (
                    "counts",
                    Value::Map(
                        [
                            (
                                MapKey::String("x".into()),
                                wrapper("Int32Value", Value::I32(7))?,
                            ),
                            (
                                MapKey::String("y".into()),
                                known(descriptor, "Int32Value", [])?,
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                ),
                ("at", timestamp(descriptor, 2, 0)?),
            ],
        )?,
        build(
            descriptor,
            "known.Known",
            [
                ("int32_value", known(descriptor, "Int32Value", [])?),
                ("string_value", known(descriptor, "StringValue", [])?),
                ("bool_value", known(descriptor, "BoolValue", [])?),
                ("created", known(descriptor, "Timestamp", [])?),
                ("attributes", json_struct(descriptor, [])?),
                (
                    "payload",
                    json_value(descriptor, "null_value", Value::EnumNumber(0))?,
                ),
                ("items", known(descriptor, "ListValue", [])?),
                ("never", Value::String("n".into())),
            ],
        )?,
    ];
    Ok(messages.iter().map(Message::encode_to_vec).collect())
}

/// [`known_messages`], decoded with the dedicated mappings.
struct KnownSample {
    /// The protobuf dtype.
    dtype: ExtDType<Protobuf>,
    /// The serialized messages.
    messages: Vec<Vec<u8>>,
    /// The decoded messages.
    array: ArrayRef,
}

impl KnownSample {
    /// Decodes [`known_messages`] into a protobuf array of the given nullability.
    fn new(nullability: Nullability) -> VortexResult<Self> {
        let dtype = protobuf_dtype_with(&[KNOWN_PROTO], "known.Known", nullability, dedicated())?;
        let messages = known_messages(dtype.metadata().message_descriptor())?;
        let rows = messages.iter().cloned().map(Some).collect::<Vec<_>>();
        let array = decode_rows(&dtype, &rows)?;
        Ok(Self {
            dtype,
            messages,
            array,
        })
    }
}

/// Each well-known type maps to its dedicated form, a `google.protobuf.Duration` stays a
/// message, and the dedicated forms keep the nullability of their position.
#[test]
fn test_well_known_storage_dtype() -> VortexResult<()> {
    let dtype = protobuf_dtype_with(
        &[KNOWN_PROTO],
        "known.Known",
        Nullability::NonNullable,
        dedicated(),
    )?;
    assert_snapshot!(describe_fields(dtype.storage_dtype()), @r"
    double_value: f64?
    float_value: f32?
    int64_value: i64?
    uint64_value: u64?
    int32_value: i32?
    uint32_value: u32?
    bool_value: bool?
    string_value: utf8?
    bytes_value: binary?
    created: vortex.timestamp[ns, tz=UTC](i64?)
    elapsed: {seconds=i64, nanos=i32}?
    attributes: vortex.json(utf8?)
    payload: vortex.json(utf8?)
    items: vortex.json(utf8?)
    history: list(vortex.timestamp[ns, tz=UTC](i64))
    counts: list({key=utf8, value=i32})
    at: vortex.timestamp[ns, tz=UTC](i64?)
    never: utf8?
    times: list({key=utf8, value=vortex.timestamp[ns, tz=UTC](i64)})
    bags: list({values=list({key=utf8, value=vortex.json(utf8)})})
    ");
    Ok(())
}

/// A message with the name of a well-known type but other fields is an ordinary message.
#[test]
fn test_well_known_needs_matching_fields() -> VortexResult<()> {
    let source = (
        "fake.proto",
        r#"
syntax = "proto3";
package google.protobuf;

message Timestamp {
  string text = 1;
}

message Holder {
  Timestamp created = 1;
}
"#,
    );
    let dtype = protobuf_dtype_with(
        &[source],
        "google.protobuf.Holder",
        Nullability::NonNullable,
        dedicated(),
    )?;
    assert_eq!(
        describe_fields(dtype.storage_dtype()),
        "created: {text=utf8}?"
    );
    Ok(())
}

/// By default the JSON types are messages, which contain themselves.
#[test]
fn test_json_messages_need_recursion_depth() -> VortexResult<()> {
    let descriptor = message(&[KNOWN_PROTO], "known.Known")?;
    let error = ProtobufMetadata::try_new(&descriptor)
        .err()
        .ok_or_else(|| vortex_err!("recursive JSON messages were accepted"))?;
    assert!(
        error.to_string().contains(
            "google.protobuf.Struct -> google.protobuf.Struct.FieldsEntry -> \
             google.protobuf.Value -> google.protobuf.Struct"
        ),
        "{error}"
    );

    let options = MappingOptions {
        max_recursion_depth: Some(0),
        ..MappingOptions::default()
    };
    let metadata = ProtobufMetadata::try_new_with_options(&descriptor, options)?;
    assert!(
        describe_fields(&metadata.plan().dtype(Nullability::NonNullable))
            .contains("created: {seconds=i64, nanos=i32}?")
    );
    Ok(())
}

/// Returns `ext_storage(protobuf_field(path, root()))`, the storage of a field stored as an
/// extension.
fn field_storage(path: &str) -> Expression {
    ext_storage(protobuf_field(path, root()))
}

/// The dedicated forms hold the values of the messages.
#[rstest]
#[case::wrapper(
    protobuf_field("int32_value", root()),
    PrimitiveArray::from_option_iter([None, Some(-4i32), Some(0)]).into_array()
)]
#[case::string_wrapper(
    protobuf_field("string_value", root()),
    VarBinViewArray::from_iter_nullable_str([None, Some("s"), Some("")]).into_array()
)]
#[case::bool_wrapper(
    protobuf_field("bool_value", root()),
    BoolArray::from_iter([None, Some(true), Some(false)]).into_array()
)]
#[case::timestamp(
    field_storage("created"),
    PrimitiveArray::from_option_iter([None, Some(1_700_000_000_000_000_123i64), Some(0)])
        .into_array()
)]
#[case::oneof_timestamp(
    field_storage("at"),
    PrimitiveArray::from_option_iter([None, Some(2_000_000_000i64), None]).into_array()
)]
#[case::json_struct(
    field_storage("attributes"),
    VarBinViewArray::from_iter_nullable_str([
        None,
        Some(r#"{"a":1.0,"b":"x\"","c":[true,null],"d":{}}"#),
        Some("{}"),
    ])
    .into_array()
)]
#[case::json_value(
    field_storage("payload"),
    VarBinViewArray::from_iter_nullable_str([None, Some("-0.5"), Some("null")]).into_array()
)]
#[case::json_list(
    field_storage("items"),
    VarBinViewArray::from_iter_nullable_str([None, Some(r#"["é"]"#), Some("[]")]).into_array()
)]
fn test_well_known_values(
    #[case] expr: Expression,
    #[case] expected: ArrayRef,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let sample = KnownSample::new(Nullability::NonNullable)?;
    assert_arrays_eq!(sample.array.apply(&expr)?, expected, &mut ctx);
    Ok(())
}

/// Lists and maps of well-known types hold non-nullable values, and an empty wrapper holds the
/// default value.
#[test]
fn test_well_known_lists_and_maps() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = KnownSample::new(Nullability::NonNullable)?.array;
    let history = array
        .clone()
        .apply(&protobuf_field("history", root()))?
        .execute_scalar(1, &mut ctx)?;
    assert_eq!(
        history.to_string(),
        "[1970-01-01T00:00:00+00:00[UTC], 1969-12-31T23:59:59.5+00:00[UTC]]"
    );
    let counts = array
        .apply(&protobuf_field("counts", root()))?
        .execute_scalar(1, &mut ctx)?;
    assert_eq!(
        counts.to_string(),
        r#"[{key: "x", value: 7i32}, {key: "y", value: 0i32}]"#
    );
    Ok(())
}

/// Messages in the dedicated forms encode back to messages that parse the same, with every
/// encoder.
#[rstest]
fn test_well_known_round_trip(
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let sample = KnownSample::new(nullability)?;
    let descriptor = sample.dtype.metadata().message_descriptor();
    let parse = |bytes: &[u8]| DynamicMessage::decode(descriptor.clone(), bytes);
    let encoded = encode_rows(&sample.array)?;
    for (row, (original, encoded)) in sample.messages.iter().zip(&encoded).enumerate() {
        let encoded = encoded
            .as_deref()
            .ok_or_else(|| vortex_err!("message {row} is null"))?;
        assert_eq!(parse(encoded)?, parse(original)?, "row {row}");

        let scalar = sample.array.execute_scalar(row, &mut ctx)?;
        let extension = scalar.as_extension();
        let storage = extension
            .value()
            .ok_or_else(|| vortex_err!("message {row} is null"))?;
        let value = Protobuf::unpack_native(&sample.dtype, storage)?;
        assert_eq!(value.encode_to_vec()?, encoded, "row {row}");
        assert_eq!(value.to_message()?, parse(original)?, "row {row}");
    }
    Ok(())
}

/// Encodes a `google.protobuf.Value` that holds a number.
fn number(value: f64) -> Vec<u8> {
    wire_field(2, WireType::SixtyFourBit, &value.to_le_bytes())
}

/// Encodes an entry of the map field `number` with a string key and a message value.
fn map_entry(number: u32, key: &str, value: &[u8]) -> Vec<u8> {
    len_field(
        number,
        &[len_field(1, key.as_bytes()), len_field(2, value)].concat(),
    )
}

/// Encodes a `google.protobuf.Struct` entry.
fn struct_entry(key: &str, value: &[u8]) -> Vec<u8> {
    map_entry(1, key, value)
}

/// The occurrences of a well-known message field merge before the message converts, so a value
/// that a later occurrence replaces need not have a stored form. Neither does a map value that a
/// later entry with the same key replaces.
#[rstest]
#[case::timestamp_parts(
    [len_field(10, &varint_field(1, 1)), len_field(10, &varint_field(2, 5))].concat(),
    field_storage("created"),
    "1000000005i64"
)]
#[case::timestamp_replaced(
    [len_field(10, &varint_field(1, 1 << 40)), len_field(10, &varint_field(1, 2))].concat(),
    field_storage("created"),
    "2000000000i64"
)]
#[case::wrapper_keeps_value(
    [len_field(5, &varint_field(1, 3)), len_field(5, &[])].concat(),
    protobuf_field("int32_value", root()),
    "3i32"
)]
#[case::wrapper_last_value(
    [len_field(5, &varint_field(1, 3)), len_field(5, &varint_field(1, 4))].concat(),
    protobuf_field("int32_value", root()),
    "4i32"
)]
#[case::json_lists_concatenate(
    [len_field(13, &len_field(6, &len_field(1, &number(1.0)))), len_field(13, &len_field(6, &len_field(1, &number(2.0))))].concat(),
    field_storage("payload"),
    "\"[1.0,2.0]\""
)]
#[case::json_last_kind_wins(
    [len_field(13, &[]), len_field(13, &number(f64::NAN)), len_field(13, &len_field(3, b"x"))].concat(),
    field_storage("payload"),
    "\"\\\"x\\\"\""
)]
#[case::json_structs_merge(
    [
        len_field(12, &struct_entry("a", &number(1.0))),
        len_field(12, &[struct_entry("b", &number(2.0)), struct_entry("a", &number(3.0))].concat()),
    ]
    .concat(),
    field_storage("attributes"),
    r#""{\"a\":3.0,\"b\":2.0}""#
)]
#[case::map_value_default(
    len_field(16, &len_field(1, b"k")),
    protobuf_field("counts", root()),
    r#"[{key: "k", value: 0i32}]"#
)]
#[case::map_value_replaced(
    [
        map_entry(19, "k", &varint_field(2, 2_000_000_000)),
        map_entry(19, "j", &varint_field(1, 3)),
        map_entry(19, "k", &varint_field(1, 5)),
    ]
    .concat(),
    protobuf_field("times", root()),
    r#"[{key: "j", value: 1970-01-01T00:00:03+00:00[UTC]}, {key: "k", value: 1970-01-01T00:00:05+00:00[UTC]}]"#
)]
#[case::nested_map_values_replaced(
    [
        len_field(20, &[map_entry(1, "k", &[]), map_entry(1, "k", &number(1.0))].concat()),
        len_field(20, &[map_entry(1, "k", &[]), map_entry(1, "k", &number(2.0))].concat()),
    ]
    .concat(),
    protobuf_field("bags", root()),
    r#"[{values: [{key: "k", value: 1.0}]}, {values: [{key: "k", value: 2.0}]}]"#
)]
fn test_well_known_merge(
    #[case] input: Vec<u8>,
    #[case] expr: Expression,
    #[case] expected: &str,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = protobuf_dtype_with(
        &[KNOWN_PROTO],
        "known.Known",
        Nullability::NonNullable,
        dedicated(),
    )?;
    let array = decode_rows(&dtype, &[Some(input)])?;
    let value = array.apply(&expr)?.execute_scalar(0, &mut ctx)?;
    assert_eq!(value.to_string(), expected);
    Ok(())
}

/// Returns a JSON value nested in `depth` lists.
fn nested_lists(depth: usize) -> Vec<u8> {
    (0..depth).fold(number(1.0), |value, _| len_field(6, &len_field(1, &value)))
}

/// A well-known message without a stored form fails as a whole message, which leaves the decoder
/// unchanged.
#[rstest]
#[case::timestamp_out_of_range(
    len_field(10, &varint_field(1, 1 << 40)),
    "outside the range of i64 nanoseconds"
)]
#[case::timestamp_negative_nanos(
    len_field(10, &varint_field(2, u64::MAX)),
    "has nanos -1 outside 0..=999999999"
)]
#[case::timestamp_in_list(
    len_field(15, &varint_field(2, 1_000_000_000)),
    "has nanos 1000000000 outside 0..=999999999"
)]
#[case::value_without_kind(len_field(13, &[]), "without a kind has no JSON representation")]
#[case::nested_value_without_kind(
    len_field(12, &struct_entry("a", &[])),
    "without a kind has no JSON representation"
)]
#[case::value_nan(len_field(13, &number(f64::NAN)), "number NaN has no JSON representation")]
#[case::value_infinity(
    len_field(14, &len_field(1, &number(f64::INFINITY))),
    "number inf has no JSON representation"
)]
#[case::value_too_deep(len_field(13, &nested_lists(101)), "nests deeper than 100 levels")]
#[case::invalid_utf8_wrapper(len_field(8, &len_field(1, &[0xff])), "Invalid UTF-8")]
#[case::invalid_utf8_json(len_field(13, &len_field(3, &[0xff])), "Invalid UTF-8")]
fn test_well_known_without_stored_form(
    #[case] input: Vec<u8>,
    #[case] expected: &str,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = protobuf_dtype_with(
        &[KNOWN_PROTO],
        "known.Known",
        Nullability::NonNullable,
        dedicated(),
    )?;
    let valid = len_field(13, &nested_lists(100));
    let mut decoder = ProtobufDecoder::new(&dtype);
    let error = decoder
        .append_message(&input)
        .err()
        .ok_or_else(|| vortex_err!("the message was decoded"))?;
    assert!(format!("{error:?}").contains(expected), "{error:?}");
    decoder.append_message(&valid)?;
    assert_arrays_eq!(
        decoder.finish()?,
        decode_rows(&dtype, &[Some(valid)])?,
        &mut ctx
    );
    Ok(())
}

/// A JSON number keeps every bit of its double through the JSON text, which holds the shortest
/// decimal form that reads back as the same double.
#[rstest]
fn test_json_numbers_round_trip(
    #[values(
        985.690_694_632_869_5,
        0.1 + 0.2,
        1.0 / 3.0,
        -2.0 / 7.0,
        1e300 / 3.0,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        -0.0
    )]
    value: f64,
) -> VortexResult<()> {
    let dtype = protobuf_dtype_with(
        &[KNOWN_PROTO],
        "known.Known",
        Nullability::NonNullable,
        dedicated(),
    )?;
    let input = len_field(13, &number(value));
    let encoded = encode_rows(&decode_rows(&dtype, &[Some(input.clone())])?)?;
    assert_eq!(encoded, [Some(input)]);
    Ok(())
}

/// Timestamps at both ends of the `i64` nanosecond range convert to their nanoseconds and back to
/// the same fields, and a nanosecond beyond either end has no stored form.
#[rstest]
#[case::min(-9_223_372_037, 145_224_192, Some(i64::MIN))]
#[case::below_min(-9_223_372_037, 145_224_191, None)]
#[case::max(9_223_372_036, 854_775_807, Some(i64::MAX))]
#[case::above_max(9_223_372_036, 854_775_808, None)]
#[case::before_epoch(-1, 999_999_999, Some(-1))]
fn test_timestamp_range(
    #[case] seconds: i64,
    #[case] nanos: u64,
    #[case] expected: Option<i64>,
) -> VortexResult<()> {
    let fields = [
        varint_field(1, seconds.cast_unsigned()),
        varint_field(2, nanos),
    ]
    .concat();
    let Some(expected) = expected else {
        let error = decode_timestamp(&fields)
            .err()
            .ok_or_else(|| vortex_err!("{seconds}s {nanos}ns was converted"))?;
        assert!(
            error
                .to_string()
                .contains("outside the range of i64 nanoseconds"),
            "{error}"
        );
        return Ok(());
    };
    assert_eq!(decode_timestamp(&fields)?, expected);
    let mut encoded = Vec::new();
    encode_timestamp(expected, &mut encoded);
    assert_eq!(encoded, fields);
    Ok(())
}

/// JSON text converts to the messages that hold it and back, and text that does not fit the
/// message type fails.
#[rstest]
#[case::value(JsonType::Value, r#"{"k":[1.5,"aé\n",null,false,{}]}"#, Ok(()))]
#[case::object(JsonType::Struct, r#"{"a":{"b":[]},"c":-0.0}"#, Ok(()))]
#[case::array(JsonType::ListValue, r#"[[],{"":true}]"#, Ok(()))]
#[case::not_an_object(JsonType::Struct, "[1]", Err("must be a JSON object"))]
#[case::not_an_array(JsonType::ListValue, "{}", Err("must be a JSON array"))]
#[case::invalid_json(JsonType::Value, "{", Err("Invalid JSON"))]
fn test_json_text_conversions(
    #[case] json_type: JsonType,
    #[case] text: &str,
    #[case] expected: Result<(), &str>,
) -> VortexResult<()> {
    let mut encoded = Vec::new();
    match (
        encode_json(json_type, text.as_bytes(), &mut encoded),
        expected,
    ) {
        (Ok(()), Ok(())) => {
            let mut decoded = Vec::new();
            decode_json(json_type, &encoded, &mut decoded)?;
            let parse = |text: &[u8]| serde_json::from_slice::<serde_json::Value>(text);
            assert_eq!(
                parse(&decoded).map_err(|err| vortex_err!("{err}"))?,
                parse(text.as_bytes()).map_err(|err| vortex_err!("{err}"))?
            );
        }
        (Err(error), Err(expected)) => {
            assert!(error.to_string().contains(expected), "{error}");
        }
        (result, expected) => {
            return Err(vortex_err!("{text}: got {result:?}, expected {expected:?}"));
        }
    }
    Ok(())
}

/// Protobuf arrays with the dedicated forms export to Arrow, with the Arrow types of the nested
/// extensions, and import back.
#[test]
fn test_well_known_arrow_round_trip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = KnownSample::new(Nullability::Nullable)?.array;
    let field = SESSION.arrow().to_arrow_field("message", array.dtype())?;
    let arrow_schema::DataType::Struct(fields) = field.data_type() else {
        return Err(vortex_err!("{field} is not a struct"));
    };
    let attributes = fields
        .iter()
        .find(|field| field.name() == "attributes")
        .ok_or_else(|| vortex_err!("no attributes field"))?;
    assert_eq!(
        attributes
            .metadata()
            .get(EXTENSION_TYPE_NAME_KEY)
            .map(String::as_str),
        Some("arrow.json")
    );

    let exported = array.clone().execute_arrow(Some(&field), &mut ctx)?;
    let imported = SESSION.arrow().from_arrow_array(exported, &field)?;
    assert_eq!(imported.dtype(), array.dtype());
    assert_arrays_eq!(imported, array, &mut ctx);
    Ok(())
}

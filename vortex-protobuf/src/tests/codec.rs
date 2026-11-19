// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoding, encoding, native values and field access on decoded messages.

use prost::Message;
use prost::bytes::Bytes;
use prost_reflect::DynamicMessage;
use prost_reflect::MapKey;
use prost_reflect::MessageDescriptor;
use prost_reflect::Value;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::expr::Expression;
use vortex_array::expr::is_not_null;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::LEGACY_PROTO;
use super::SESSION;
use super::all_types_dtype;
use super::build;
use super::protobuf_dtype;
use crate::Protobuf;
use crate::ProtobufDecoder;
use crate::decode_messages;
use crate::encode_messages;
use crate::protobuf_field;

/// Builds a message value of type `name` from the pool of `descriptor` with the given fields.
fn build_value(
    descriptor: &MessageDescriptor,
    name: &str,
    fields: impl IntoIterator<Item = (&'static str, Value)>,
) -> VortexResult<Value> {
    build(descriptor, name, fields).map(Value::Message)
}

/// Builds a `test.Inner` message value with the given fields.
fn inner(
    descriptor: &MessageDescriptor,
    fields: impl IntoIterator<Item = (&'static str, Value)>,
) -> VortexResult<Value> {
    build_value(descriptor, "test.Inner", fields)
}

/// Returns a `Value::String`.
fn string(value: &str) -> Value {
    Value::String(value.to_owned())
}

/// Serialized `test.AllTypes` messages that cover absent fields, empty lists, default values and
/// presence.
pub(super) fn sample_messages(descriptor: &MessageDescriptor) -> VortexResult<Vec<Vec<u8>>> {
    let messages = [
        // Every field is absent.
        build(descriptor, "test.AllTypes", [])?,
        // Every field holds a value that is not its default.
        build(
            descriptor,
            "test.AllTypes",
            [
                ("f_double", Value::F64(1.5)),
                ("f_float", Value::F32(-2.5)),
                ("f_int32", Value::I32(-3)),
                ("f_int64", Value::I64(-4)),
                ("f_uint32", Value::U32(5)),
                ("f_uint64", Value::U64(u64::MAX)),
                ("f_sint32", Value::I32(-7)),
                ("f_sint64", Value::I64(i64::MIN)),
                ("f_fixed32", Value::U32(9)),
                ("f_fixed64", Value::U64(10)),
                ("f_sfixed32", Value::I32(-11)),
                ("f_sfixed64", Value::I64(-12)),
                ("f_bool", Value::Bool(true)),
                ("f_string", string("hello")),
                ("f_bytes", Value::Bytes(Bytes::from_static(&[0, 1, 2]))),
                ("f_enum", Value::EnumNumber(1)),
                (
                    "inner",
                    inner(
                        descriptor,
                        [("name", string("a")), ("count", Value::I64(1))],
                    )?,
                ),
                (
                    "outer",
                    build_value(
                        descriptor,
                        "test.Outer",
                        [
                            ("inner", inner(descriptor, [("name", string("b"))])?),
                            (
                                "items",
                                Value::List(vec![
                                    inner(descriptor, [("count", Value::I64(2))])?,
                                    inner(descriptor, [])?,
                                ]),
                            ),
                        ],
                    )?,
                ),
                (
                    "r_int32",
                    Value::List(vec![Value::I32(1), Value::I32(0), Value::I32(-1)]),
                ),
                ("r_string", Value::List(vec![string("x"), string("")])),
                (
                    "r_inner",
                    Value::List(vec![inner(descriptor, [("name", string("c"))])?]),
                ),
                (
                    "m_inner",
                    Value::Map(
                        [
                            (
                                MapKey::String("k1".to_owned()),
                                inner(descriptor, [("count", Value::I64(3))])?,
                            ),
                            (MapKey::String("k0".to_owned()), inner(descriptor, [])?),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                ),
                (
                    "m_labels",
                    Value::Map(
                        [
                            (MapKey::I32(2), string("two")),
                            (MapKey::I32(-1), string("")),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                ),
                ("c_inner", inner(descriptor, [("name", string("d"))])?),
                ("o_int32", Value::I32(5)),
                ("o_string", string("opt")),
                (
                    "created",
                    build_value(
                        descriptor,
                        "google.protobuf.Timestamp",
                        [("seconds", Value::I64(100)), ("nanos", Value::I32(5))],
                    )?,
                ),
            ],
        )?,
        // Fields with explicit presence hold default values, and an empty message is present.
        build(
            descriptor,
            "test.AllTypes",
            [
                ("inner", inner(descriptor, [])?),
                ("outer", build_value(descriptor, "test.Outer", [])?),
                ("c_text", string("")),
                ("o_int32", Value::I32(0)),
                ("o_string", string("")),
                ("r_int32", Value::List(vec![])),
            ],
        )?,
        // Fields with implicit presence hold default values, and the enum number is unknown.
        build(
            descriptor,
            "test.AllTypes",
            [
                ("f_int32", Value::I32(0)),
                ("f_string", string("")),
                ("f_enum", Value::EnumNumber(7)),
                ("c_text", string("text")),
            ],
        )?,
    ];
    Ok(messages.iter().map(Message::encode_to_vec).collect())
}

/// Serialized `legacy.Legacy` messages that cover absent fields with explicit presence, a custom
/// default, a missing required field and a group.
fn legacy_messages(descriptor: &MessageDescriptor) -> VortexResult<Vec<Vec<u8>>> {
    let messages = [
        // Every field is absent, including the required `b` and `c`, whose default is 7.
        build(descriptor, "legacy.Legacy", [])?,
        // Every field holds a value, and `c` holds its default.
        build(
            descriptor,
            "legacy.Legacy",
            [
                ("a", Value::I32(1)),
                ("b", string("x")),
                ("c", Value::I64(7)),
                ("d", Value::List(vec![Value::U32(1), Value::U32(2)])),
                (
                    "settings",
                    build_value(
                        descriptor,
                        "legacy.Legacy.Settings",
                        [("flag", Value::Bool(true))],
                    )?,
                ),
            ],
        )?,
        // Present fields hold zero values, and the group is present but empty.
        build(
            descriptor,
            "legacy.Legacy",
            [
                ("a", Value::I32(0)),
                ("b", string("")),
                ("c", Value::I64(0)),
                (
                    "settings",
                    build_value(descriptor, "legacy.Legacy.Settings", [])?,
                ),
            ],
        )?,
    ];
    Ok(messages.iter().map(Message::encode_to_vec).collect())
}

/// Parses serialized `bytes` as a message of type `descriptor`.
fn parse(descriptor: &MessageDescriptor, bytes: &[u8]) -> VortexResult<DynamicMessage> {
    Ok(DynamicMessage::decode(descriptor.clone(), bytes)?)
}

/// Converts the message at `row` of the protobuf array `array` of `dtype` to a protobuf message.
fn native_message(
    dtype: &ExtDType<Protobuf>,
    array: &ArrayRef,
    row: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DynamicMessage> {
    let scalar = array.execute_scalar(row, ctx)?;
    let extension = scalar.as_extension();
    let storage = extension
        .value()
        .ok_or_else(|| vortex_err!("message {row} is null"))?;
    Protobuf::unpack_native(dtype, storage)?.to_message()
}

/// Returns the bytes of every row of a binary array that holds no nulls.
fn encoded_messages(array: &VarBinViewArray) -> VortexResult<Vec<ByteBuffer>> {
    binary_values(array)?
        .into_iter()
        .map(|bytes| bytes.ok_or_else(|| vortex_err!("unexpected null message")))
        .collect()
}

/// Returns the values of a binary array, or `None` at null rows.
fn binary_values(array: &VarBinViewArray) -> VortexResult<Vec<Option<ByteBuffer>>> {
    let validity = array
        .validity()?
        .execute_mask(array.len(), &mut SESSION.create_execution_ctx())?;
    Ok((0..array.len())
        .map(|row| validity.value(row).then(|| array.bytes_at(row)))
        .collect())
}

#[test]
fn test_round_trip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = all_types_dtype(Nullability::NonNullable)?;
    let descriptor = dtype.metadata().message_descriptor();
    let messages = sample_messages(descriptor)?;

    let array = decode_messages(&dtype, messages.iter().map(Vec::as_slice))?;
    assert_eq!(array.len(), messages.len());

    let encoded = encoded_messages(&encode_messages(&array, &mut ctx)?)?;
    for (row, (original, encoded)) in messages.iter().zip(&encoded).enumerate() {
        assert_eq!(
            parse(descriptor, encoded)?,
            parse(descriptor, original)?,
            "message {row}"
        );
    }

    let decoded_again = decode_messages(&dtype, encoded.iter().map(ByteBuffer::as_slice))?;
    assert_arrays_eq!(decoded_again, array, &mut ctx);
    Ok(())
}

#[test]
fn test_round_trip_null_messages() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = all_types_dtype(Nullability::Nullable)?;
    let descriptor = dtype.metadata().message_descriptor();
    let messages = sample_messages(descriptor)?;

    let mut decoder = ProtobufDecoder::new(&dtype);
    decoder.append_null()?;
    decoder.append_message(&messages[1])?;
    decoder.append_null()?;
    decoder.append_message(&messages[0])?;
    assert_eq!(decoder.len(), 4);
    let array = decoder.finish()?;

    let encoded = binary_values(&encode_messages(&array, &mut ctx)?)?;
    assert!(encoded[0].is_none());
    assert!(encoded[2].is_none());
    for (row, original) in [(1, &messages[1]), (3, &messages[0])] {
        let encoded = encoded[row]
            .as_ref()
            .ok_or_else(|| vortex_err!("message {row} is null"))?;
        assert_eq!(parse(descriptor, encoded)?, parse(descriptor, original)?);
    }

    let null_inner = protobuf_field("inner.name", root());
    assert_arrays_eq!(
        array.apply(&null_inner)?,
        VarBinViewArray::from_iter_nullable_str([None, Some("a"), None, None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_append_null_requires_nullable_dtype() -> VortexResult<()> {
    let dtype = all_types_dtype(Nullability::NonNullable)?;
    let error = ProtobufDecoder::new(&dtype)
        .append_null()
        .err()
        .ok_or_else(|| vortex_err!("a null message was appended"))?;
    assert!(
        error
            .to_string()
            .contains("Cannot append a null message to a non-nullable protobuf array"),
        "{error}"
    );
    Ok(())
}

#[test]
fn test_append_message_rejects_invalid_bytes() -> VortexResult<()> {
    let dtype = all_types_dtype(Nullability::NonNullable)?;
    let mut decoder = ProtobufDecoder::new(&dtype);
    // Field 14 (`f_string`) with a length that runs past the end of the input.
    let error = decoder
        .append_message(&[0x72, 0x05, b'a'])
        .err()
        .ok_or_else(|| vortex_err!("invalid bytes were decoded"))?;
    assert!(
        error
            .to_string()
            .contains("Invalid serialized test.AllTypes message"),
        "{error}"
    );
    assert!(decoder.is_empty());
    // The same field with the length of its value decodes.
    decoder.append_message(&[0x72, 0x01, b'a'])?;
    assert_eq!(decoder.len(), 1);
    Ok(())
}

#[rstest]
#[case::implicit_scalar("f_int32", PrimitiveArray::from_iter([0i32, -3, 0, 0]).into_array())]
#[case::implicit_string(
    "f_string",
    VarBinViewArray::from_iter_str(["", "hello", "", ""]).into_array()
)]
#[case::enum_number("f_enum", PrimitiveArray::from_iter([0i32, 1, 0, 7]).into_array())]
#[case::optional_scalar(
    "o_int32",
    PrimitiveArray::from_option_iter([None, Some(5i32), Some(0), None]).into_array()
)]
#[case::optional_string(
    "o_string",
    VarBinViewArray::from_iter_nullable_str([None, Some("opt"), Some(""), None]).into_array()
)]
#[case::oneof_scalar(
    "c_text",
    VarBinViewArray::from_iter_nullable_str([None, None, Some(""), Some("text")]).into_array()
)]
#[case::oneof_message(
    "c_inner.name",
    VarBinViewArray::from_iter_nullable_str([None, Some("d"), None, None]).into_array()
)]
#[case::nested_message(
    "inner.name",
    VarBinViewArray::from_iter_nullable_str([None, Some("a"), Some(""), None]).into_array()
)]
#[case::doubly_nested_message(
    "outer.inner.name",
    VarBinViewArray::from_iter_nullable_str([None, Some("b"), None, None]).into_array()
)]
#[case::well_known_type(
    "created.seconds",
    PrimitiveArray::from_option_iter([None, Some(100i64), None, None]).into_array()
)]
fn test_protobuf_field(#[case] path: &str, #[case] expected: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = all_types_dtype(Nullability::NonNullable)?;
    let messages = sample_messages(dtype.metadata().message_descriptor())?;
    let array = decode_messages(&dtype, messages.iter().map(Vec::as_slice))?;

    assert_arrays_eq!(
        array.apply(&protobuf_field(path, root()))?,
        expected,
        &mut ctx
    );
    Ok(())
}

/// A path fails to bind when a segment names no field, or names a field inside a repeated field,
/// while a path that differs only in that respect binds.
#[rstest]
#[case::missing_field(
    "inner.missing",
    "inner.name",
    "Couldn't find the missing field in the input scope"
)]
#[case::field_in_repeated_field(
    "r_inner.name",
    "inner.name",
    "Couldn't find the name field in the input scope"
)]
fn test_protobuf_field_rejects_invalid_paths(
    #[case] path: &str,
    #[case] valid_path: &str,
    #[case] expected: &str,
) -> VortexResult<()> {
    let dtype = all_types_dtype(Nullability::NonNullable)?;
    let messages = sample_messages(dtype.metadata().message_descriptor())?;
    let array = decode_messages(&dtype, messages.iter().map(Vec::as_slice))?;

    array.clone().apply(&protobuf_field(valid_path, root()))?;
    let error = array
        .apply(&protobuf_field(path, root()))
        .err()
        .ok_or_else(|| vortex_err!("{path} was bound"))?;
    assert!(error.to_string().contains(expected), "{error}");
    Ok(())
}

#[test]
fn test_map_entries_are_ordered_by_key() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = all_types_dtype(Nullability::NonNullable)?;
    let messages = sample_messages(dtype.metadata().message_descriptor())?;
    let array = decode_messages(&dtype, messages.iter().map(Vec::as_slice))?;

    let labels = array
        .apply(&protobuf_field("m_labels", root()))?
        .execute_scalar(1, &mut ctx)?;
    assert_eq!(
        labels.to_string(),
        r#"[{key: -1i32, value: ""}, {key: 2i32, value: "two"}]"#
    );
    Ok(())
}

#[test]
fn test_native_value_converts_to_message() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = all_types_dtype(Nullability::NonNullable)?;
    let descriptor = dtype.metadata().message_descriptor();
    let messages = sample_messages(descriptor)?;
    let array = decode_messages(&dtype, messages.iter().map(Vec::as_slice))?;

    for (row, original) in messages.iter().enumerate() {
        let message = native_message(&dtype, &array, row, &mut ctx)?;
        assert_eq!(message, parse(descriptor, original)?);
        assert_eq!(
            parse(descriptor, &message.encode_to_vec())?,
            parse(descriptor, original)?
        );
    }
    Ok(())
}

/// A `test.AllTypes` message whose `double` and `float` fields with implicit presence hold `-0.0`.
///
/// `prost-reflect` does not write `-0.0`, which equals the default zero, so the message sets
/// `f_double` (field 1, 64-bit) and `f_float` (field 2, 32-bit) by hand.
fn negative_zero_message() -> Vec<u8> {
    [
        [0x09].as_slice(),
        &(-0.0f64).to_le_bytes(),
        &[0x15],
        &(-0.0f32).to_le_bytes(),
    ]
    .concat()
}

/// A `double` or `float` field with implicit presence keeps a `-0.0` from the wire, which
/// equals its default zero.
#[test]
fn test_decode_keeps_negative_zero() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = all_types_dtype(Nullability::NonNullable)?;
    let descriptor = dtype.metadata().message_descriptor();
    let bytes = negative_zero_message();
    let array = decode_messages(&dtype, [bytes.as_slice()])?;

    let f_double = array
        .clone()
        .apply(&protobuf_field("f_double", root()))?
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert!(f_double.as_slice::<f64>()[0].is_sign_negative());
    let f_float = array
        .clone()
        .apply(&protobuf_field("f_float", root()))?
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert!(f_float.as_slice::<f32>()[0].is_sign_negative());
    assert_eq!(
        native_message(&dtype, &array, 0, &mut ctx)?,
        parse(descriptor, &bytes)?
    );
    Ok(())
}

/// The encoders write a `-0.0` of a field with implicit presence, so that it survives a round
/// trip.
#[test]
fn test_encode_keeps_negative_zero() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = all_types_dtype(Nullability::NonNullable)?;
    let bytes = negative_zero_message();
    let array = decode_messages(&dtype, [bytes.as_slice()])?;

    let encoded = encoded_messages(&encode_messages(&array, &mut ctx)?)?;
    assert_eq!(encoded[0].as_slice(), bytes.as_slice());
    let scalar = array.execute_scalar(0, &mut ctx)?;
    let extension = scalar.as_extension();
    let storage = extension
        .value()
        .ok_or_else(|| vortex_err!("message 0 is null"))?;
    assert_eq!(
        Protobuf::unpack_native(&dtype, storage)?.encode_to_vec()?,
        bytes
    );
    Ok(())
}

/// Proto2 fields with explicit presence are null when absent, even when the schema declares a
/// default value or the field is required, and a group is stored as a message.
#[rstest]
#[case::optional_scalar(
    protobuf_field("a", root()),
    Ok(PrimitiveArray::from_option_iter([None, Some(1i32), Some(0)]).into_array())
)]
#[case::required_string(
    protobuf_field("b", root()),
    Ok(VarBinViewArray::from_iter_nullable_str([None, Some("x"), Some("")]).into_array())
)]
#[case::custom_default(
    protobuf_field("c", root()),
    Ok(PrimitiveArray::from_option_iter([None, Some(7i64), Some(0)]).into_array())
)]
#[case::repeated_scalar(
    protobuf_field("d", root()),
    ListArray::try_new(
        buffer![1u32, 2].into_array(),
        buffer![0u32, 0, 2, 2].into_array(),
        Validity::NonNullable,
        &mut SESSION.create_execution_ctx(),
    )
    .map(IntoArray::into_array)
)]
#[case::group_presence(
    is_not_null(protobuf_field("settings", root())),
    Ok(BoolArray::from_iter([false, true, true]).into_array())
)]
#[case::group_field(
    protobuf_field("settings.flag", root()),
    Ok(BoolArray::from_iter([None, Some(true), None]).into_array())
)]
fn test_proto2_field(
    #[case] expr: Expression,
    #[case] expected: VortexResult<ArrayRef>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = protobuf_dtype(&[LEGACY_PROTO], "legacy.Legacy", Nullability::NonNullable)?;
    let messages = legacy_messages(dtype.metadata().message_descriptor())?;
    let array = decode_messages(&dtype, messages.iter().map(Vec::as_slice))?;

    assert_arrays_eq!(array.apply(&expr)?, expected?, &mut ctx);
    Ok(())
}

/// Proto2 messages encode back to their original bytes, with the group between its start and end
/// group tags, and convert back to the messages that the bytes hold.
#[test]
fn test_proto2_round_trip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = protobuf_dtype(&[LEGACY_PROTO], "legacy.Legacy", Nullability::NonNullable)?;
    let descriptor = dtype.metadata().message_descriptor();
    // The messages are built with the compiled descriptor, not the one of the dtype metadata.
    let messages = legacy_messages(&super::message(&[LEGACY_PROTO], "legacy.Legacy")?)?;
    let array = decode_messages(&dtype, messages.iter().map(Vec::as_slice))?;

    let encoded = encoded_messages(&encode_messages(&array, &mut ctx)?)?;
    for (row, (original, encoded)) in messages.iter().zip(&encoded).enumerate() {
        assert_eq!(encoded.as_slice(), original.as_slice(), "message {row}");
        assert_eq!(
            native_message(&dtype, &array, row, &mut ctx)?,
            parse(descriptor, original)?,
            "message {row}"
        );
    }
    // Field 5 starts and ends the group with wire types 3 and 4.
    assert!(encoded[1].as_slice().ends_with(&[0x2b, 0x30, 0x01, 0x2c]));
    Ok(())
}

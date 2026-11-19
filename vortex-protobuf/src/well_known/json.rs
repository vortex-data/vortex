// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversions between the JSON messages of `google.protobuf` and JSON text, following the
//! proto3 JSON mapping.
//!
//! A `Value` holds one kind of JSON value in a oneof, a `Struct` holds a JSON object as a map
//! from keys to values, and a `ListValue` holds a JSON array. Decoding follows protobuf parse
//! semantics: the last kind of a `Value` wins, a `struct_value` or `list_value` that occurs again
//! while it is the kind merges into the earlier one, and a later map entry replaces an earlier
//! entry with the same key.

use std::collections::BTreeMap;

use prost::encoding::DecodeContext;
use prost::encoding::WireType;
use prost::encoding::decode_key;
use prost::encoding::encode_key;
use prost::encoding::encode_varint;
use prost::encoding::skip_field;
use serde_json::Map;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use super::JsonType;
use crate::wire::read_fixed64;
use crate::wire::read_length_delimited;
use crate::wire::read_varint;

/// The deepest nesting of JSON values that decoding accepts, the nesting limit of protobuf
/// parsers.
const NESTING_LIMIT: usize = 100;

/// A decoded JSON value, with strings borrowed from the encoding.
enum Node<'a> {
    /// `null`.
    Null,
    /// A number.
    Number(f64),
    /// A string.
    String(&'a str),
    /// `true` or `false`.
    Bool(bool),
    /// An object, whose values are `None` where a `Value` has no kind.
    Object(BTreeMap<&'a str, Option<Node<'a>>>),
    /// An array, whose elements are `None` where a `Value` has no kind.
    Array(Vec<Option<Node<'a>>>),
}

/// Converts an encoded message of `json_type` to JSON text, appended to `out`.
///
/// # Errors
///
/// Returns an error if `input` is not a valid encoding, if a `Value` has no kind, if a number is
/// a NaN or an infinity, or if the values nest deeper than protobuf parsers allow.
pub(crate) fn decode_json(
    json_type: JsonType,
    input: &[u8],
    out: &mut Vec<u8>,
) -> VortexResult<()> {
    match json_type {
        JsonType::Value => {
            let mut value = None;
            merge_value(&mut value, input, 0)?;
            write_text(value.as_ref(), out)
        }
        JsonType::Struct => {
            let mut fields = BTreeMap::new();
            merge_struct(&mut fields, input, 0)?;
            write_text(Some(&Node::Object(fields)), out)
        }
        JsonType::ListValue => {
            let mut values = Vec::new();
            merge_list(&mut values, input, 0)?;
            write_text(Some(&Node::Array(values)), out)
        }
    }
}

/// Merges the fields of an encoded `Value` into `value`, which is `None` without a kind.
fn merge_value<'a>(
    value: &mut Option<Node<'a>>,
    mut input: &'a [u8],
    depth: usize,
) -> VortexResult<()> {
    if depth > NESTING_LIMIT {
        vortex_bail!("A google.protobuf.Value nests deeper than {NESTING_LIMIT} levels");
    }
    while !input.is_empty() {
        let (number, wire_type) = decode_key(&mut input)?;
        match (number, wire_type) {
            (1, WireType::Varint) => {
                read_varint(&mut input)?;
                *value = Some(Node::Null);
            }
            (2, WireType::SixtyFourBit) => {
                *value = Some(Node::Number(f64::from_bits(read_fixed64(&mut input)?)));
            }
            (3, WireType::LengthDelimited) => {
                let bytes = read_length_delimited(&mut input)?;
                let string = simdutf8::basic::from_utf8(bytes)
                    .map_err(|_| vortex_err!("Invalid UTF-8 in a google.protobuf.Value string"))?;
                *value = Some(Node::String(string));
            }
            (4, WireType::Varint) => *value = Some(Node::Bool(read_varint(&mut input)? != 0)),
            (5, WireType::LengthDelimited) => {
                let payload = read_length_delimited(&mut input)?;
                match value {
                    Some(Node::Object(fields)) => merge_struct(fields, payload, depth + 1)?,
                    _ => {
                        let mut fields = BTreeMap::new();
                        merge_struct(&mut fields, payload, depth + 1)?;
                        *value = Some(Node::Object(fields));
                    }
                }
            }
            (6, WireType::LengthDelimited) => {
                let payload = read_length_delimited(&mut input)?;
                match value {
                    Some(Node::Array(values)) => merge_list(values, payload, depth + 1)?,
                    _ => {
                        let mut values = Vec::new();
                        merge_list(&mut values, payload, depth + 1)?;
                        *value = Some(Node::Array(values));
                    }
                }
            }
            _ => skip_field(wire_type, number, &mut input, DecodeContext::default())?,
        }
    }
    Ok(())
}

/// Merges the entries of an encoded `Struct` into `fields`.
fn merge_struct<'a>(
    fields: &mut BTreeMap<&'a str, Option<Node<'a>>>,
    mut input: &'a [u8],
    depth: usize,
) -> VortexResult<()> {
    while !input.is_empty() {
        let (number, wire_type) = decode_key(&mut input)?;
        if (number, wire_type) != (1, WireType::LengthDelimited) {
            skip_field(wire_type, number, &mut input, DecodeContext::default())?;
            continue;
        }
        let mut entry = read_length_delimited(&mut input)?;
        let (mut key, mut value) = ("", None);
        while !entry.is_empty() {
            let (number, wire_type) = decode_key(&mut entry)?;
            match (number, wire_type) {
                (1, WireType::LengthDelimited) => {
                    key = simdutf8::basic::from_utf8(read_length_delimited(&mut entry)?).map_err(
                        |_| vortex_err!("Invalid UTF-8 in a google.protobuf.Struct key"),
                    )?;
                }
                (2, WireType::LengthDelimited) => {
                    merge_value(&mut value, read_length_delimited(&mut entry)?, depth)?;
                }
                _ => skip_field(wire_type, number, &mut entry, DecodeContext::default())?,
            }
        }
        fields.insert(key, value);
    }
    Ok(())
}

/// Appends the values of an encoded `ListValue` to `values`.
fn merge_list<'a>(
    values: &mut Vec<Option<Node<'a>>>,
    mut input: &'a [u8],
    depth: usize,
) -> VortexResult<()> {
    while !input.is_empty() {
        let (number, wire_type) = decode_key(&mut input)?;
        if (number, wire_type) != (1, WireType::LengthDelimited) {
            skip_field(wire_type, number, &mut input, DecodeContext::default())?;
            continue;
        }
        let mut value = None;
        merge_value(&mut value, read_length_delimited(&mut input)?, depth)?;
        values.push(value);
    }
    Ok(())
}

/// Writes `value` as JSON text, with object keys in sorted order.
fn write_text(value: Option<&Node<'_>>, out: &mut Vec<u8>) -> VortexResult<()> {
    let Some(value) = value else {
        vortex_bail!("A google.protobuf.Value without a kind has no JSON representation");
    };
    match value {
        Node::Null => out.extend_from_slice(b"null"),
        Node::Bool(value) => out.extend_from_slice(if *value { b"true" } else { b"false" }),
        Node::Number(number) => {
            if !number.is_finite() {
                vortex_bail!("A google.protobuf.Value number {number} has no JSON representation");
            }
            serde_json::to_writer(&mut *out, number).map_err(json_error)?;
        }
        Node::String(string) => serde_json::to_writer(&mut *out, string).map_err(json_error)?,
        Node::Object(fields) => {
            out.push(b'{');
            for (index, (key, value)) in fields.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                serde_json::to_writer(&mut *out, key).map_err(json_error)?;
                out.push(b':');
                write_text(value.as_ref(), out)?;
            }
            out.push(b'}');
        }
        Node::Array(values) => {
            out.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                write_text(value.as_ref(), out)?;
            }
            out.push(b']');
        }
    }
    Ok(())
}

/// Converts an error of writing a JSON string or number.
fn json_error(err: serde_json::Error) -> VortexError {
    vortex_err!("Cannot write JSON: {err}")
}

/// Writes the fields of a message of `json_type` that holds the JSON `text`.
///
/// A number becomes the double nearest to it, so a number that [`decode_json`] wrote becomes the
/// double that it came from. That needs the `float_roundtrip` feature of `serde_json`, whose
/// default parser can be off by one unit in the last place.
///
/// # Errors
///
/// Returns an error if `text` is not valid JSON, or not an object for a `Struct` or an array for
/// a `ListValue`.
pub(crate) fn encode_json(json_type: JsonType, text: &[u8], buf: &mut Vec<u8>) -> VortexResult<()> {
    let value = serde_json::from_slice::<serde_json::Value>(text)
        .map_err(|err| vortex_err!("Invalid JSON for a google.protobuf message: {err}"))?;
    match (json_type, &value) {
        (JsonType::Value, value) => write_value(value, buf),
        (JsonType::Struct, serde_json::Value::Object(fields)) => write_struct(fields, buf),
        (JsonType::ListValue, serde_json::Value::Array(values)) => write_list(values, buf),
        (JsonType::Struct, _) => vortex_bail!("A google.protobuf.Struct must be a JSON object"),
        (JsonType::ListValue, _) => {
            vortex_bail!("A google.protobuf.ListValue must be a JSON array")
        }
    }
}

/// Writes the fields of a `Value` that holds `value`.
fn write_value(value: &serde_json::Value, buf: &mut Vec<u8>) -> VortexResult<()> {
    match value {
        serde_json::Value::Null => {
            encode_key(1, WireType::Varint, buf);
            encode_varint(0, buf);
        }
        serde_json::Value::Number(number) => {
            let number = number
                .as_f64()
                .ok_or_else(|| vortex_err!("JSON number {number} is not a double"))?;
            encode_key(2, WireType::SixtyFourBit, buf);
            buf.extend_from_slice(&number.to_le_bytes());
        }
        serde_json::Value::String(string) => write_bytes(3, string.as_bytes(), buf),
        serde_json::Value::Bool(value) => {
            encode_key(4, WireType::Varint, buf);
            encode_varint(u64::from(*value), buf);
        }
        serde_json::Value::Object(fields) => {
            write_nested(5, buf, |nested| write_struct(fields, nested))?;
        }
        serde_json::Value::Array(values) => {
            write_nested(6, buf, |nested| write_list(values, nested))?;
        }
    }
    Ok(())
}

/// Writes the fields of a `Struct` that holds `fields`, one map entry per key.
fn write_struct(fields: &Map<String, serde_json::Value>, buf: &mut Vec<u8>) -> VortexResult<()> {
    for (key, value) in fields {
        write_nested(1, buf, |entry| {
            write_bytes(1, key.as_bytes(), entry);
            write_nested(2, entry, |nested| write_value(value, nested))
        })?;
    }
    Ok(())
}

/// Writes the fields of a `ListValue` that holds `values`.
fn write_list(values: &[serde_json::Value], buf: &mut Vec<u8>) -> VortexResult<()> {
    for value in values {
        write_nested(1, buf, |nested| write_value(value, nested))?;
    }
    Ok(())
}

/// Writes a length-delimited field `number` that holds `bytes`.
fn write_bytes(number: u32, bytes: &[u8], buf: &mut Vec<u8>) {
    encode_key(number, WireType::LengthDelimited, buf);
    encode_varint(bytes.len() as u64, buf);
    buf.extend_from_slice(bytes);
}

/// Writes a length-delimited field `number` that holds the message that `write` writes.
fn write_nested(
    number: u32,
    buf: &mut Vec<u8>,
    write: impl FnOnce(&mut Vec<u8>) -> VortexResult<()>,
) -> VortexResult<()> {
    let mut nested = Vec::new();
    write(&mut nested)?;
    write_bytes(number, &nested, buf);
    Ok(())
}

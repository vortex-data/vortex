// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Malformed, truncated and corrupted input, which fails without changing the decoder.

use prost::encoding::DecodeContext;
use prost::encoding::WireType;
use prost::encoding::decode_key;
use prost::encoding::decode_varint;
use prost::encoding::skip_field;
use prost_reflect::DynamicMessage;
use prost_reflect::Kind;
use prost_reflect::MessageDescriptor;
use rstest::rstest;
use vortex_array::VortexSessionExecute;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::SESSION;
use super::differential::ALL_TYPES;
use super::differential::CLOSED;
use super::differential::LEGACY;
use super::differential::REPEATED;
use super::differential::Sample;
use super::differential::Schema;
use super::len_field;
use super::preserving;
use super::varint_field;
use super::wire_field;
use crate::MappingOptions;
use crate::ProtobufDecoder;
use crate::decode_messages;

/// Each input fails to decode with an error that names the problem, and leaves the decoder
/// unchanged, so that a valid message still decodes after it.
#[rstest]
#[case::truncated_varint(ALL_TYPES, vec![0x18, 0x80], "invalid varint")]
#[case::overlong_varint(
    ALL_TYPES,
    [[0x18].as_slice(), &[0xff; 10], &[0x01]].concat(),
    "invalid varint"
)]
#[case::truncated_fixed64(ALL_TYPES, vec![0x09, 1, 2, 3], "a 64-bit value needs 8 bytes")]
#[case::truncated_fixed32(ALL_TYPES, vec![0x15, 1], "a 32-bit value needs 4 bytes")]
#[case::length_past_end(
    ALL_TYPES,
    vec![0x72, 0x05, b'a'],
    "a length-delimited value of 5 bytes exceeds the 1 remaining bytes"
)]
#[case::length_past_nested_end(
    ALL_TYPES,
    len_field(17, &[0x0a, 0x05, b'a']),
    "a length-delimited value of 5 bytes exceeds the 1 remaining bytes"
)]
#[case::invalid_wire_type_6(ALL_TYPES, vec![0x1e, 0x00], "invalid wire type value: 6")]
#[case::invalid_wire_type_7(ALL_TYPES, vec![0x1f], "invalid wire type value: 7")]
#[case::field_number_zero(ALL_TYPES, vec![0x00, 0x00], "invalid tag value: 0")]
#[case::unexpected_end_group(
    ALL_TYPES,
    wire_field(3, WireType::EndGroup, &[]),
    "Unexpected protobuf end group tag of field 3"
)]
#[case::invalid_utf8(ALL_TYPES, len_field(14, &[b'a', 0xff]), "Invalid UTF-8")]
#[case::invalid_utf8_in_map_key(
    ALL_TYPES,
    len_field(22, &len_field(1, &[0xc3])),
    "Invalid UTF-8"
)]
#[case::invalid_utf8_in_map_value(
    ALL_TYPES,
    len_field(23, &len_field(2, &[0xc3])),
    "Invalid UTF-8"
)]
#[case::truncated_packed_varint(ALL_TYPES, len_field(19, &[0x01, 0x80]), "invalid varint")]
#[case::partial_packed_fixed32(
    REPEATED,
    len_field(8, &[1, 2, 3, 4, 5]),
    "a 32-bit value needs 4 bytes"
)]
#[case::truncated_map_entry(
    ALL_TYPES,
    len_field(22, &[0x0a, 0x03, b'k']),
    "a length-delimited value of 3 bytes exceeds the 1 remaining bytes"
)]
#[case::unterminated_group(
    LEGACY,
    [wire_field(5, WireType::StartGroup, &[]), varint_field(6, 1)].concat(),
    "group field 5 has no end group tag"
)]
#[case::mismatched_end_group(
    LEGACY,
    [wire_field(5, WireType::StartGroup, &[]), wire_field(6, WireType::EndGroup, &[])].concat(),
    "Unexpected protobuf end group tag of field 6"
)]
#[case::unterminated_unknown_group(
    ALL_TYPES,
    [wire_field(99, WireType::StartGroup, &[]), varint_field(1, 1)].concat(),
    "invalid varint"
)]
#[case::unknown_length_past_end(ALL_TYPES, vec![0x9a, 0x06, 0x0a], "buffer underflow")]
fn test_decode_rejects_malformed_input(
    #[case] schema: Schema,
    #[case] input: Vec<u8>,
    #[case] expected: &str,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let sample = Sample::new(schema, Nullability::NonNullable)?;
    let valid = sample.canonical();

    let mut decoder = ProtobufDecoder::new(&sample.dtype);
    let message = sample.dtype.metadata().message_name().to_owned();
    for valid in valid.iter().flatten().take(3) {
        decoder.append_message(valid)?;
        let error = decoder
            .append_message(&input)
            .err()
            .ok_or_else(|| vortex_err!("the malformed input decoded"))?
            .to_string();
        assert!(error.contains(expected), "{error}");
        assert!(
            error.contains(&format!("Invalid serialized {message} message")),
            "{error}"
        );
    }
    assert_eq!(decoder.len(), 3);
    assert!(
        DynamicMessage::decode(sample.descriptor().clone(), input.as_slice()).is_err(),
        "prost-reflect decodes the malformed input"
    );
    assert_arrays_eq!(decoder.finish()?, sample.decode(&valid[..3])?, &mut ctx);
    Ok(())
}

/// A message that fails after many of its fields were decoded leaves no values behind: the
/// decoder rolls back every column, including lists, maps, nested messages and unknown fields.
#[rstest]
fn test_decode_rolls_back_failed_messages(
    #[values(ALL_TYPES, REPEATED, LEGACY, CLOSED)] schema: Schema,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
    #[values(MappingOptions::default(), preserving())] options: MappingOptions,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut sample = Sample::with_options(schema, nullability, options)?;
    let valid = sample.non_canonical();

    let mut decoder = ProtobufDecoder::new(&sample.dtype);
    for row in &valid {
        let Some(message) = row else {
            decoder.append_null()?;
            continue;
        };
        // An invalid wire type after all fields of a message.
        let invalid = [message.as_slice(), &[0x0f]].concat();
        assert!(decoder.append_message(&invalid).is_err());
        decoder.append_message(message)?;
    }
    assert_arrays_eq!(decoder.finish()?, sample.decode(&valid)?, &mut ctx);
    Ok(())
}

/// Every prefix of an encoding decodes if and only if `prost-reflect` parses it, and with the
/// default options, which the oracle knows, to the same values. A rejected prefix leaves no trace,
/// so the decoder ends with the array of the accepted ones.
#[rstest]
fn test_decode_truncated_messages(
    #[values(ALL_TYPES, REPEATED, LEGACY, CLOSED)] schema: Schema,
    #[values(MappingOptions::default(), preserving())] options: MappingOptions,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut sample = Sample::with_options(schema, Nullability::NonNullable, options)?;
    let descriptor = sample.descriptor().clone();

    let mut decoder = ProtobufDecoder::new(&sample.dtype);
    let mut accepted = Vec::new();
    for message in sample.non_canonical().into_iter().flatten().take(40) {
        for end in 0..message.len() {
            let prefix = &message[..end];
            let parsed = DynamicMessage::decode(descriptor.clone(), prefix).is_ok();
            assert_eq!(
                decoder.append_message(prefix).is_ok(),
                parsed,
                "prefix of {end} bytes of {message:02x?}"
            );
            if parsed {
                accepted.push(Some(prefix.to_vec()));
            }
        }
    }
    assert_eq!(decoder.len(), accepted.len());
    let decoded = decoder.finish()?;
    assert_arrays_eq!(decoded, sample.decode(&accepted)?, &mut ctx);
    if sample.has_oracle() {
        assert_arrays_eq!(decoded, sample.oracle(&accepted)?, &mut ctx);
    }
    Ok(())
}

/// Corrupted encodings never panic. A rejected one leaves no trace, so the decoder ends with the
/// array of the accepted ones. A corruption whose known fields all have the wire types of their
/// fields decodes if and only if `prost-reflect` parses it, and with the default options to the
/// same values. Closed enums are left out of that comparison, because `prost-reflect` keeps
/// numbers that a closed enum does not define.
#[rstest]
fn test_decode_corrupted_messages(
    #[values(ALL_TYPES, REPEATED, LEGACY, CLOSED)] schema: Schema,
    #[values(MappingOptions::default(), preserving())] options: MappingOptions,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut sample = Sample::with_options(schema, Nullability::NonNullable, options)?;
    let descriptor = sample.descriptor().clone();
    let messages = sample.non_canonical();

    let mut decoder = ProtobufDecoder::new(&sample.dtype);
    let mut accepted = Vec::new();
    let mut parsed = Vec::new();
    for message in messages.into_iter().flatten() {
        for _ in 0..10 {
            let corrupted = corrupt(&mut sample, &message);
            let decoded = decoder.append_message(&corrupted).is_ok();
            if schema != CLOSED && wire_types_match(&descriptor, &mut corrupted.as_slice(), None) {
                let is_parsed =
                    DynamicMessage::decode(descriptor.clone(), corrupted.as_slice()).is_ok();
                assert_eq!(decoded, is_parsed, "{corrupted:02x?}");
                if is_parsed {
                    parsed.push(Some(corrupted.clone()));
                }
            }
            if decoded {
                accepted.push(corrupted);
            }
        }
    }
    assert!(!accepted.is_empty());
    assert!(schema == CLOSED || !parsed.is_empty());
    assert_eq!(decoder.len(), accepted.len());
    assert_arrays_eq!(
        decoder.finish()?,
        decode_messages(&sample.dtype, accepted.iter().map(Vec::as_slice))?,
        &mut ctx
    );
    if sample.has_oracle() {
        assert_arrays_eq!(sample.decode(&parsed)?, sample.oracle(&parsed)?, &mut ctx);
    }
    Ok(())
}

/// Whether every occurrence of a known field in `input`, nested messages and map entries
/// included, has a wire type that the field accepts, and the input is well-formed enough to tell.
///
/// Protobuf parsers skip an occurrence with another wire type like an unknown field.
/// `prost-reflect` rejects it instead, except in a map field, whose wire type it does not check.
fn wire_types_match(descriptor: &MessageDescriptor, input: &mut &[u8], group: Option<u32>) -> bool {
    while !input.is_empty() {
        let Ok((number, wire_type)) = decode_key(input) else {
            return false;
        };
        if wire_type == WireType::EndGroup {
            return group == Some(number);
        }
        let Some(field) = descriptor.get_field(number) else {
            if skip_field(wire_type, number, input, DecodeContext::default()).is_err() {
                return false;
            }
            continue;
        };
        let kind = field.kind();
        let packed = field.is_list()
            && wire_type == WireType::LengthDelimited
            && !matches!(kind, Kind::String | Kind::Bytes | Kind::Message(_));
        let expected = if field.is_group() {
            WireType::StartGroup
        } else {
            kind.wire_type()
        };
        if wire_type != expected && !packed {
            return false;
        }
        let nested = match &kind {
            Kind::Message(message) if field.is_group() => {
                wire_types_match(message, input, Some(number))
            }
            Kind::Message(message) => decode_varint(input)
                .ok()
                .and_then(|len| usize::try_from(len).ok())
                .filter(|&len| len <= input.len())
                .is_some_and(|len| {
                    let (payload, rest) = input.split_at(len);
                    *input = rest;
                    wire_types_match(message, &mut &payload[..], None)
                }),
            _ => skip_field(wire_type, number, input, DecodeContext::default()).is_ok(),
        };
        if !nested {
            return false;
        }
    }
    group.is_none()
}

/// Returns `message` with a random corruption: changed, inserted or removed bytes.
fn corrupt(sample: &mut Sample, message: &[u8]) -> Vec<u8> {
    let generator = &mut sample.generator;
    let mut corrupted = message.to_vec();
    for _ in 0..=generator.below(3) {
        let position = generator.below(corrupted.len() + 1);
        match generator.below(4) {
            0 if position < corrupted.len() => {
                corrupted[position] ^= 1 << generator.below(8);
            }
            1 => {
                let byte = [0x00, 0x7f, 0x80, 0xff, 0x08, 0x0a, 0x0b, 0x0c][generator.below(8)];
                corrupted.insert(position, byte);
            }
            2 if position < corrupted.len() => {
                corrupted.remove(position);
            }
            _ => corrupted.truncate(position),
        }
    }
    corrupted
}

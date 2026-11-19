// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Differential tests of the codec against `prost-reflect`, on seeded random messages.

use prost::Message;
use prost_reflect::DynamicMessage;
use prost_reflect::MessageDescriptor;
use prost_reflect::Value;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::CLOSED_PROTO;
use super::LEGACY_PROTO;
use super::REPEATED_PROTO;
use super::SESSION;
use super::TEST_PROTO;
use super::UNRELATED_PROTO;
use super::generate::Generator;
use super::oracle::oracle_decode;
use super::protobuf_dtype_with;
use super::well_known::KNOWN_PROTO;
use super::well_known::dedicated;
use crate::JsonMapping;
use crate::MappingOptions;
use crate::Protobuf;
use crate::ProtobufDecoder;
use crate::TimestampMapping;
use crate::encode_messages;

/// The number of messages that each test generates.
const MESSAGES: usize = 200;

/// A schema that the tests generate messages for: its sources and its root message.
pub(super) type Schema = (&'static [(&'static str, &'static str)], &'static str);

/// `test.AllTypes`: every scalar kind, nested messages, repeated fields, maps, a oneof, proto3
/// `optional` fields and a well-known type.
pub(super) const ALL_TYPES: Schema = (&[TEST_PROTO, UNRELATED_PROTO], "test.AllTypes");

/// `repeated.Repeated`: packed and unpacked repeated fields of every scalar kind, maps with every
/// kind of key and a large field number.
pub(super) const REPEATED: Schema = (&[REPEATED_PROTO], "repeated.Repeated");

/// `legacy.Legacy`: proto2 with explicit presence, a custom default and a group.
pub(super) const LEGACY: Schema = (&[LEGACY_PROTO], "legacy.Legacy");

/// `closed.Closed`: proto2 with a closed enum, a repeated group and packed fields.
pub(super) const CLOSED: Schema = (&[CLOSED_PROTO], "closed.Closed");

/// `known.Known`: every well-known type that has a dedicated form, also repeated, in maps and in a
/// oneof.
pub(super) const KNOWN: Schema = (&[KNOWN_PROTO], "known.Known");

/// Random messages of a schema, as a nullable or non-nullable protobuf dtype.
pub(super) struct Sample {
    /// The protobuf dtype.
    pub(super) dtype: ExtDType<Protobuf>,
    /// The messages, or `None` at null rows, which only a nullable dtype has.
    pub(super) messages: Vec<Option<DynamicMessage>>,
    /// The generator, for further random choices.
    pub(super) generator: Generator,
}

impl Sample {
    /// Generates the messages of `schema` from a seed that depends on the schema, for the default
    /// mapping options.
    pub(super) fn new(schema: Schema, nullability: Nullability) -> VortexResult<Self> {
        Self::with_options(schema, nullability, MappingOptions::default())
    }

    /// Generates the messages of `schema` from a seed that depends on the schema, for the mapping
    /// `options`.
    ///
    /// If the options store timestamps or JSON in their dedicated forms, the messages hold only
    /// values that these forms store.
    pub(super) fn with_options(
        schema: Schema,
        nullability: Nullability,
        options: MappingOptions,
    ) -> VortexResult<Self> {
        let (sources, name) = schema;
        let dedicated_forms = options.timestamps == TimestampMapping::Nanoseconds
            || options.json == JsonMapping::Text;
        let dtype = protobuf_dtype_with(sources, name, nullability, options)?;
        let seed = name
            .bytes()
            .fold(17u64, |seed, byte| seed.wrapping_mul(31) + u64::from(byte));
        let mut generator = Generator::new(seed);
        if dedicated_forms {
            generator = generator.for_dedicated_forms();
        }
        let descriptor = dtype.metadata().message_descriptor().clone();
        let messages = (0..MESSAGES)
            .map(|_| {
                (!nullability.is_nullable() || generator.chance(0.8))
                    .then(|| generator.message(&descriptor))
            })
            .collect();
        Ok(Self {
            dtype,
            messages,
            generator,
        })
    }

    /// The message type.
    pub(super) fn descriptor(&self) -> &MessageDescriptor {
        self.dtype.metadata().message_descriptor()
    }

    /// Encodes each message with `prost-reflect`.
    pub(super) fn canonical(&self) -> Vec<Option<Vec<u8>>> {
        self.messages
            .iter()
            .map(|message| message.as_ref().map(Message::encode_to_vec))
            .collect()
    }

    /// Encodes each message in a random non-canonical form.
    pub(super) fn non_canonical(&mut self) -> Vec<Option<Vec<u8>>> {
        let generator = &mut self.generator;
        self.messages
            .iter()
            .map(|message| {
                message
                    .as_ref()
                    .map(|message| generator.non_canonical(message))
            })
            .collect()
    }

    /// Decodes `rows` with the decoder under test.
    pub(super) fn decode(&self, rows: &[Option<Vec<u8>>]) -> VortexResult<ArrayRef> {
        let mut decoder = ProtobufDecoder::new(&self.dtype);
        for row in rows {
            match row {
                Some(message) => decoder.append_message(message)?,
                None => decoder.append_null()?,
            }
        }
        decoder.finish()
    }

    /// Whether the `prost-reflect` oracle knows the mapping of the dtype, which it does only for
    /// the default options.
    pub(super) fn has_oracle(&self) -> bool {
        self.dtype.metadata().options() == &MappingOptions::default()
    }

    /// Decodes `rows` with the `prost-reflect` oracle, which needs [`Self::has_oracle`].
    pub(super) fn oracle(&self, rows: &[Option<Vec<u8>>]) -> VortexResult<ArrayRef> {
        oracle_decode(&self.dtype, rows.iter().map(Option::as_deref))
    }
}

/// Parses serialized `bytes` as a message of type `descriptor` with `prost-reflect`.
fn parse(descriptor: &MessageDescriptor, bytes: &[u8]) -> VortexResult<DynamicMessage> {
    Ok(DynamicMessage::decode(descriptor.clone(), bytes)?)
}

/// Returns the values of a binary array, or `None` at null rows.
fn binary_values(array: &VarBinViewArray) -> VortexResult<Vec<Option<Vec<u8>>>> {
    let validity = array
        .validity()?
        .execute_mask(array.len(), &mut SESSION.create_execution_ctx())?;
    Ok((0..array.len())
        .map(|row| {
            validity
                .value(row)
                .then(|| array.bytes_at(row).as_slice().to_vec())
        })
        .collect())
}

/// Whether every map field of `message` and of its nested messages holds at most one entry, so
/// that `prost-reflect`, which writes map entries in hash order, writes them in a known order.
fn maps_have_one_entry(message: &DynamicMessage) -> bool {
    message
        .fields()
        .all(|(_, value)| value_maps_have_one_entry(value))
}

/// [`maps_have_one_entry`] for any value.
fn value_maps_have_one_entry(value: &Value) -> bool {
    match value {
        Value::Message(message) => maps_have_one_entry(message),
        Value::List(values) => values.iter().all(value_maps_have_one_entry),
        Value::Map(entries) => {
            entries.len() <= 1 && entries.values().all(value_maps_have_one_entry)
        }
        _ => true,
    }
}

/// The decoder builds the same arrays as the `prost-reflect` oracle from canonical encodings.
#[rstest]
fn test_decode_matches_oracle(
    #[values(ALL_TYPES, REPEATED, LEGACY, CLOSED)] schema: Schema,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let sample = Sample::new(schema, nullability)?;
    let canonical = sample.canonical();
    assert_arrays_eq!(
        sample.decode(&canonical)?,
        sample.oracle(&canonical)?,
        &mut ctx
    );
    Ok(())
}

/// Non-canonical encodings of the messages decode to the same arrays as the canonical ones,
/// with both decoders.
#[rstest]
fn test_decode_non_canonical_encodings(
    #[values(ALL_TYPES, REPEATED, LEGACY, CLOSED)] schema: Schema,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut sample = Sample::new(schema, nullability)?;
    let canonical = sample.canonical();
    let non_canonical = sample.non_canonical();
    assert_ne!(canonical, non_canonical);

    let expected = sample.decode(&canonical)?;
    assert_arrays_eq!(sample.oracle(&non_canonical)?, expected, &mut ctx);
    assert_arrays_eq!(sample.decode(&non_canonical)?, expected, &mut ctx);
    Ok(())
}

/// A concatenation of two encodings decodes as the merge of the two messages, as `prost-reflect`
/// parses it.
#[rstest]
fn test_decode_concatenated_messages(
    #[values(ALL_TYPES, REPEATED, LEGACY, CLOSED)] schema: Schema,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut sample = Sample::new(schema, Nullability::NonNullable)?;
    let canonical = sample.canonical();
    let non_canonical = sample.non_canonical();
    let concatenated = canonical
        .iter()
        .zip(non_canonical.iter().cycle().skip(1))
        .map(|(first, second)| {
            let mut bytes = first.clone().unwrap_or_default();
            bytes.extend_from_slice(second.as_deref().unwrap_or_default());
            Some(bytes)
        })
        .collect::<Vec<_>>();
    assert_arrays_eq!(
        sample.decode(&concatenated)?,
        sample.oracle(&concatenated)?,
        &mut ctx
    );
    Ok(())
}

/// Encoding decoded messages gives bytes that decode to the same arrays and parse to the same
/// messages. Without maps of several entries, which `prost-reflect` writes in hash order, the
/// bytes equal those of `prost-reflect`. The scalar values encode to the same bytes.
#[rstest]
fn test_encode_round_trip(
    #[values(ALL_TYPES, REPEATED, LEGACY, CLOSED)] schema: Schema,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let sample = Sample::new(schema, nullability)?;
    let canonical = sample.canonical();
    let decoded = sample.decode(&canonical)?;

    let encoded = binary_values(&encode_messages(&decoded, &mut ctx)?)?;
    assert_arrays_eq!(sample.decode(&encoded)?, decoded, &mut ctx);

    let descriptor = sample.descriptor();
    for (row, (message, encoded)) in sample.messages.iter().zip(&encoded).enumerate() {
        let (Some(message), Some(encoded)) = (message, encoded) else {
            assert_eq!(message.is_none(), encoded.is_none(), "row {row}");
            continue;
        };
        let original = message.encode_to_vec();
        assert_eq!(
            parse(descriptor, encoded)?,
            parse(descriptor, &original)?,
            "row {row}"
        );
        if maps_have_one_entry(message) {
            assert_eq!(encoded, &original, "row {row}");
        }

        let scalar = decoded.execute_scalar(row, &mut ctx)?;
        let extension = scalar.as_extension();
        let storage = extension
            .value()
            .ok_or_else(|| vortex_err!("row {row} is null"))?;
        assert_eq!(
            &Protobuf::unpack_native(&sample.dtype, storage)?.encode_to_vec()?,
            encoded,
            "row {row}"
        );
    }
    Ok(())
}

/// With the dedicated forms of the well-known types, random messages decode to the same arrays
/// from their canonical and non-canonical encodings, and encode back to messages that parse the
/// same, as arrays and as scalars.
///
/// The messages hold full-precision JSON numbers and timestamps up to the ends of the nanosecond
/// range. Their non-canonical encodings hold values without a stored form, such as timestamps
/// out of range and JSON values without a kind, where a later occurrence, oneof member or map
/// entry with the same key replaces them.
#[rstest]
fn test_dedicated_forms_round_trip(
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut sample = Sample::with_options(KNOWN, nullability, dedicated())?;
    let canonical = sample.canonical();
    let non_canonical = sample.non_canonical();
    let decoded = sample.decode(&canonical)?;
    assert_arrays_eq!(sample.decode(&non_canonical)?, decoded, &mut ctx);

    let encoded = binary_values(&encode_messages(&decoded, &mut ctx)?)?;
    assert_arrays_eq!(sample.decode(&encoded)?, decoded, &mut ctx);
    let descriptor = sample.descriptor();
    for (row, (original, encoded)) in canonical.iter().zip(&encoded).enumerate() {
        let (Some(original), Some(encoded)) = (original, encoded) else {
            assert_eq!(original.is_none(), encoded.is_none(), "row {row}");
            continue;
        };
        assert_eq!(
            parse(descriptor, encoded)?,
            parse(descriptor, original)?,
            "row {row}"
        );

        let scalar = decoded.execute_scalar(row, &mut ctx)?;
        let extension = scalar.as_extension();
        let storage = extension
            .value()
            .ok_or_else(|| vortex_err!("row {row} is null"))?;
        assert_eq!(
            &Protobuf::unpack_native(&sample.dtype, storage)?.encode_to_vec()?,
            encoded,
            "row {row}"
        );
    }
    Ok(())
}

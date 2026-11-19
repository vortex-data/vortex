// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `google.protobuf` well-known types that the mapping options store in a dedicated form.
//!
//! The decoder collects the encoding of such a message, all its occurrences in one message, and
//! converts it once the message is complete. The encoder converts each value back to an encoding.

mod json;

use prost::encoding::DecodeContext;
use prost::encoding::WireType;
use prost::encoding::decode_key;
use prost::encoding::encode_key;
use prost::encoding::encode_varint;
use prost::encoding::skip_field;
use prost_reflect::FieldDescriptor;
use prost_reflect::Kind;
use prost_reflect::MessageDescriptor;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_json::Json;

use self::FieldType::Named;
use self::FieldType::Repeated;
use self::FieldType::Scalar;
use self::FieldType::StringMap;
pub(crate) use self::json::decode_json;
pub(crate) use self::json::encode_json;
use crate::JsonMapping;
use crate::MappingOptions;
use crate::TimestampMapping;
use crate::WrapperMapping;
use crate::mapping::scalar_dtype;
use crate::wire::ScalarKind;
use crate::wire::read_varint;
use crate::wire::varint_to_i32;

/// The time zone of stored timestamps, which are instants.
const TIMESTAMP_ZONE: &str = "UTC";

/// The nanoseconds of a second.
const NANOS_PER_SECOND: i64 = 1_000_000_000;

/// A well-known message type that the mapping options store in a dedicated form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WellKnown {
    /// A wrapper message, stored as its wrapped value of this kind.
    Wrapper(ScalarKind),
    /// `google.protobuf.Timestamp`, stored as a `vortex.timestamp` in nanoseconds.
    Timestamp,
    /// `google.protobuf.Value`, `Struct` or `ListValue`, stored as `vortex.json` text.
    Json(JsonType),
}

/// The `google.protobuf` messages that hold JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JsonType {
    /// `google.protobuf.Value`, any JSON value.
    Value,
    /// `google.protobuf.Struct`, a JSON object.
    Struct,
    /// `google.protobuf.ListValue`, a JSON array.
    ListValue,
}

impl WellKnown {
    /// Returns the dedicated form that `options` choose for messages of type `message`, or `None`
    /// if they are stored as structs.
    ///
    /// A message is a well-known type if it has the full name and the fields of one.
    pub(crate) fn of(message: &MessageDescriptor, options: &MappingOptions) -> Option<Self> {
        let (well_known, fields): (Self, &[FieldType]) = match message.full_name() {
            "google.protobuf.DoubleValue" => (
                Self::Wrapper(ScalarKind::Double),
                &[Scalar(ScalarKind::Double)],
            ),
            "google.protobuf.FloatValue" => (
                Self::Wrapper(ScalarKind::Float),
                &[Scalar(ScalarKind::Float)],
            ),
            "google.protobuf.Int64Value" => (
                Self::Wrapper(ScalarKind::Int64),
                &[Scalar(ScalarKind::Int64)],
            ),
            "google.protobuf.UInt64Value" => (
                Self::Wrapper(ScalarKind::Uint64),
                &[Scalar(ScalarKind::Uint64)],
            ),
            "google.protobuf.Int32Value" => (
                Self::Wrapper(ScalarKind::Int32),
                &[Scalar(ScalarKind::Int32)],
            ),
            "google.protobuf.UInt32Value" => (
                Self::Wrapper(ScalarKind::Uint32),
                &[Scalar(ScalarKind::Uint32)],
            ),
            "google.protobuf.BoolValue" => {
                (Self::Wrapper(ScalarKind::Bool), &[Scalar(ScalarKind::Bool)])
            }
            "google.protobuf.StringValue" => (
                Self::Wrapper(ScalarKind::String),
                &[Scalar(ScalarKind::String)],
            ),
            "google.protobuf.BytesValue" => (
                Self::Wrapper(ScalarKind::Bytes),
                &[Scalar(ScalarKind::Bytes)],
            ),
            "google.protobuf.Timestamp" => (
                Self::Timestamp,
                &[Scalar(ScalarKind::Int64), Scalar(ScalarKind::Int32)],
            ),
            "google.protobuf.Value" => (
                Self::Json(JsonType::Value),
                &[
                    Named("google.protobuf.NullValue"),
                    Scalar(ScalarKind::Double),
                    Scalar(ScalarKind::String),
                    Scalar(ScalarKind::Bool),
                    Named("google.protobuf.Struct"),
                    Named("google.protobuf.ListValue"),
                ],
            ),
            "google.protobuf.Struct" => (
                Self::Json(JsonType::Struct),
                &[StringMap("google.protobuf.Value")],
            ),
            "google.protobuf.ListValue" => (
                Self::Json(JsonType::ListValue),
                &[Repeated("google.protobuf.Value")],
            ),
            _ => return None,
        };
        let chosen = match well_known {
            Self::Wrapper(_) => options.wrappers == WrapperMapping::Primitive,
            Self::Timestamp => options.timestamps == TimestampMapping::Nanoseconds,
            Self::Json(_) => options.json == JsonMapping::Text,
        };
        // The fields of each well-known type are numbered from 1 in declaration order.
        let matches = message.fields().len() == fields.len()
            && message
                .fields()
                .zip(fields)
                .zip(1..)
                .all(|((field, field_type), number)| {
                    field.number() == number && field_type.matches(&field)
                });
        (chosen && matches).then_some(well_known)
    }

    /// The storage dtype of a value with the given nullability.
    pub(crate) fn dtype(self, nullability: Nullability) -> DType {
        match self {
            Self::Wrapper(kind) => scalar_dtype(kind, nullability),
            Self::Timestamp | Self::Json(_) => DType::Extension(self.ext_dtype(nullability)),
        }
    }

    /// The extension dtype of a timestamp or JSON value with the given nullability.
    ///
    /// # Panics
    ///
    /// Panics for a wrapper, which is not stored as an extension.
    pub(crate) fn ext_dtype(self, nullability: Nullability) -> ExtDTypeRef {
        match self {
            Self::Timestamp => Timestamp::new_with_tz(
                TimeUnit::Nanoseconds,
                Some(TIMESTAMP_ZONE.into()),
                nullability,
            )
            .erased(),
            Self::Json(_) => ExtDType::<Json>::try_new(EmptyMetadata, nullability)
                .vortex_expect("JSON storage has a non-nullable form")
                .erased(),
            Self::Wrapper(kind) => {
                vortex_panic!("Protobuf {kind:?} wrappers are not stored as extensions")
            }
        }
    }
}

/// The type of a field of a well-known message.
#[derive(Clone, Copy)]
enum FieldType {
    /// A singular scalar of this kind.
    Scalar(ScalarKind),
    /// A singular enum or message of the type with this full name.
    Named(&'static str),
    /// A repeated message of the type with this full name.
    Repeated(&'static str),
    /// A map from strings to messages of the type with this full name.
    StringMap(&'static str),
}

impl FieldType {
    /// Whether `field` has this type.
    fn matches(self, field: &FieldDescriptor) -> bool {
        let kind = field.kind();
        let singular = !field.is_list() && !field.is_map();
        match self {
            Self::Scalar(scalar) => singular && ScalarKind::of(&kind) == Some(scalar),
            Self::Named(name) => singular && type_name(&kind) == Some(name),
            Self::Repeated(name) => field.is_list() && type_name(&kind) == Some(name),
            Self::StringMap(name) => {
                field.is_map()
                    && kind.as_message().is_some_and(|entry| {
                        matches!(entry.map_entry_key_field().kind(), Kind::String)
                            && type_name(&entry.map_entry_value_field().kind()) == Some(name)
                    })
            }
        }
    }
}

/// Returns the full name of the enum or message type of `kind`, or `None` for a scalar.
fn type_name(kind: &Kind) -> Option<&str> {
    match kind {
        Kind::Message(message) => Some(message.full_name()),
        Kind::Enum(enum_type) => Some(enum_type.full_name()),
        _ => None,
    }
}

/// Returns the encoding of the last `value` field of a wrapper message of `kind`, if any.
///
/// # Errors
///
/// Returns an error if `input` is not a valid message encoding.
pub(crate) fn wrapper_value(kind: ScalarKind, mut input: &[u8]) -> VortexResult<Option<&[u8]>> {
    let mut value = None;
    while !input.is_empty() {
        let (number, wire_type) = decode_key(&mut input)?;
        let start = input;
        skip_field(wire_type, number, &mut input, DecodeContext::default())?;
        if number == 1 && wire_type == kind.wire_type() {
            value = Some(&start[..start.len() - input.len()]);
        }
    }
    Ok(value)
}

/// Reads an encoded `google.protobuf.Timestamp` and returns its nanoseconds since the Unix epoch.
///
/// # Errors
///
/// Returns an error if `input` is not a valid encoding, if `nanos` is outside
/// `0..=999_999_999`, or if the instant is outside the `i64` nanosecond range.
pub(crate) fn decode_timestamp(mut input: &[u8]) -> VortexResult<i64> {
    let (mut seconds, mut nanos) = (0i64, 0i32);
    while !input.is_empty() {
        let (number, wire_type) = decode_key(&mut input)?;
        match (number, wire_type) {
            (1, WireType::Varint) => seconds = read_varint(&mut input)?.cast_signed(),
            (2, WireType::Varint) => nanos = varint_to_i32(read_varint(&mut input)?),
            _ => skip_field(wire_type, number, &mut input, DecodeContext::default())?,
        }
    }
    vortex_ensure!(
        (0..NANOS_PER_SECOND).contains(&i64::from(nanos)),
        "A google.protobuf.Timestamp has nanos {nanos} outside 0..=999999999"
    );
    // In i128, because the seconds of an instant in the first second of the i64 range overflow
    // i64 when they are multiplied out.
    let total = i128::from(seconds) * i128::from(NANOS_PER_SECOND) + i128::from(nanos);
    i64::try_from(total).map_err(|_| {
        vortex_err!(
            "A google.protobuf.Timestamp of {seconds} seconds is outside the range of i64 \
                 nanoseconds"
        )
    })
}

/// Writes the fields of a `google.protobuf.Timestamp` that is `nanos` nanoseconds after the Unix
/// epoch, leaving out zero fields.
pub(crate) fn encode_timestamp(nanos: i64, buf: &mut Vec<u8>) {
    let seconds = nanos.div_euclid(NANOS_PER_SECOND);
    let nanos = nanos.rem_euclid(NANOS_PER_SECOND);
    if seconds != 0 {
        encode_key(1, WireType::Varint, buf);
        encode_varint(seconds.cast_unsigned(), buf);
    }
    if nanos != 0 {
        encode_key(2, WireType::Varint, buf);
        encode_varint(nanos.cast_unsigned(), buf);
    }
}

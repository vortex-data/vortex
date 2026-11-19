// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The choices of the storage mapping that the metadata of a protobuf dtype records.

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

/// The largest [`max_recursion_depth`](MappingOptions::max_recursion_depth), which is the default
/// nesting limit of protobuf parsers.
pub const MAX_RECURSION_DEPTH: u32 = 100;

/// The choices of the storage mapping from a message schema to a Vortex storage dtype.
///
/// The options are part of the [`ProtobufMetadata`](crate::ProtobufMetadata), so the storage
/// dtype follows from the metadata alone. The defaults store well-known types as ordinary
/// messages, which hold every value, drop unknown fields and reject recursive messages. The crate
/// documentation lists what a round trip keeps under each option.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MappingOptions {
    /// Stores the fields that the schema does not define, with values whose wire type does not
    /// match their field, in a `$unknown` binary field of each message, and writes them back when
    /// encoding. So messages of a newer schema keep their new fields.
    pub preserve_unknown_fields: bool,
    /// Accepts recursive messages, which contain themselves directly or through other messages.
    ///
    /// A message is stored as a struct where it occurs inside itself at most this many times, and
    /// a deeper occurrence is stored as the binary encoding of the message. `None` rejects
    /// recursive messages. The depth can be at most [`MAX_RECURSION_DEPTH`].
    pub max_recursion_depth: Option<u32>,
    /// How to store the `google.protobuf` wrapper messages, such as `Int32Value`.
    pub wrappers: WrapperMapping,
    /// How to store `google.protobuf.Timestamp` messages.
    pub timestamps: TimestampMapping,
    /// How to store the `google.protobuf.Struct`, `Value` and `ListValue` messages.
    pub json: JsonMapping,
}

/// How to store the `google.protobuf` wrapper messages: `DoubleValue`, `FloatValue`,
/// `Int64Value`, `UInt64Value`, `Int32Value`, `UInt32Value`, `BoolValue`, `StringValue` and
/// `BytesValue`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WrapperMapping {
    /// As ordinary messages, structs with one `value` field.
    #[default]
    Message,
    /// As the wrapped value, which is null where the field is absent.
    ///
    /// Unknown fields inside the wrapper messages are dropped.
    Primitive,
}

/// How to store `google.protobuf.Timestamp` messages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TimestampMapping {
    /// As ordinary messages, structs of `seconds` and `nanos`, which hold every value.
    #[default]
    Message,
    /// As the `vortex.timestamp` extension in nanoseconds in UTC.
    ///
    /// Decoding fails for an instant outside the `i64` nanosecond range, from 1677-09-21 to
    /// 2262-04-11, and for `nanos` outside `0..=999_999_999`. Unknown fields inside the timestamp
    /// messages are dropped.
    Nanoseconds,
}

/// How to store the `google.protobuf.Struct`, `Value` and `ListValue` messages, which hold JSON.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum JsonMapping {
    /// As ordinary messages. `Value` contains itself through `Struct` and `ListValue`, so this
    /// needs [`max_recursion_depth`](MappingOptions::max_recursion_depth).
    #[default]
    Message,
    /// As the `vortex.json` extension, with the JSON text of the proto3 JSON mapping.
    ///
    /// Decoding fails for a `Value` without a kind and for a number that JSON cannot represent,
    /// a NaN or an infinity. Object keys are written in sorted order, and unknown fields inside
    /// these messages are dropped.
    Text,
}

/// The serialized form of [`MappingOptions`].
///
/// An option that changes the storage dtype needs a new mapping version, because a reader of an
/// older version would ignore its unknown field.
#[derive(Clone, PartialEq, Message)]
pub(crate) struct SerializedOptions {
    /// See [`MappingOptions::preserve_unknown_fields`].
    #[prost(bool, tag = "1")]
    pub(crate) preserve_unknown_fields: bool,
    /// See [`MappingOptions::max_recursion_depth`].
    #[prost(uint32, optional, tag = "2")]
    pub(crate) max_recursion_depth: Option<u32>,
    /// See [`MappingOptions::wrappers`]: 0 for messages and 1 for primitives.
    #[prost(uint32, tag = "3")]
    pub(crate) wrappers: u32,
    /// See [`MappingOptions::timestamps`]: 0 for messages and 1 for nanoseconds.
    #[prost(uint32, tag = "4")]
    pub(crate) timestamps: u32,
    /// See [`MappingOptions::json`]: 0 for messages and 1 for JSON text.
    #[prost(uint32, tag = "5")]
    pub(crate) json: u32,
}

impl MappingOptions {
    /// Checks that the options are in range.
    pub(crate) fn validate(&self) -> VortexResult<()> {
        if let Some(depth) = self.max_recursion_depth {
            vortex_ensure!(
                depth <= MAX_RECURSION_DEPTH,
                "Protobuf max_recursion_depth {depth} exceeds the limit of {MAX_RECURSION_DEPTH}"
            );
        }
        Ok(())
    }

    /// Converts the options to their serialized form.
    pub(crate) fn serialize(&self) -> SerializedOptions {
        SerializedOptions {
            preserve_unknown_fields: self.preserve_unknown_fields,
            max_recursion_depth: self.max_recursion_depth,
            wrappers: match self.wrappers {
                WrapperMapping::Message => 0,
                WrapperMapping::Primitive => 1,
            },
            timestamps: match self.timestamps {
                TimestampMapping::Message => 0,
                TimestampMapping::Nanoseconds => 1,
            },
            json: match self.json {
                JsonMapping::Message => 0,
                JsonMapping::Text => 1,
            },
        }
    }

    /// Converts serialized options back.
    ///
    /// # Errors
    ///
    /// Returns an error if an option holds a value that this crate does not know.
    pub(crate) fn deserialize(serialized: &SerializedOptions) -> VortexResult<Self> {
        let options = Self {
            preserve_unknown_fields: serialized.preserve_unknown_fields,
            max_recursion_depth: serialized.max_recursion_depth,
            wrappers: match serialized.wrappers {
                0 => WrapperMapping::Message,
                1 => WrapperMapping::Primitive,
                other => vortex_bail!("Unknown protobuf wrapper mapping {other}"),
            },
            timestamps: match serialized.timestamps {
                0 => TimestampMapping::Message,
                1 => TimestampMapping::Nanoseconds,
                other => vortex_bail!("Unknown protobuf timestamp mapping {other}"),
            },
            json: match serialized.json {
                0 => JsonMapping::Message,
                1 => JsonMapping::Text,
                other => vortex_bail!("Unknown protobuf JSON mapping {other}"),
            },
        };
        options.validate()?;
        Ok(options)
    }
}

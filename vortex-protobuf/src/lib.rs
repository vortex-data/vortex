// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]

//! An experimental extension type that stores protobuf messages of one schema in a Vortex column.
//!
//! The [`Protobuf`] dtype, with ID `vortex.protobuf`, keeps the message schema and the
//! [`MappingOptions`] in its [`ProtobufMetadata`], and stores each message as a struct with one
//! field per message field. Query engines read single fields through the storage without decoding
//! whole messages. The [`protobuf_field`] expression reads a field by its dotted path,
//! [`protobuf_enum_name`] reads the names of enum values, and [`protobuf_oneof_case`] reads which
//! member of a oneof is set.
//!
//! [`ProtobufDecoder`] builds arrays from serialized messages, and [`encode_messages`] serializes
//! them again. Both work on the wire format and the storage columns directly, without a message
//! object per row, and follow protobuf parse semantics. [`initialize`] registers the dtype, the
//! [`EnumName`] function and JSON support with a session.
//!
//! # Storage mapping
//!
//! The storage dtype is a function of the metadata: the message descriptor, the mapping version
//! and the mapping options. This crate writes [`MAPPING_VERSION`] 2, which has options. It also
//! reads version 1, which has no options and maps every schema as version 2 does with the
//! default options. The mapping is:
//!
//! - A message is a struct whose fields have the protobuf field names, in declaration order,
//!   followed by a [`UNKNOWN_FIELDS`] field if the options preserve unknown fields.
//! - A field with explicit presence is nullable, and an absent field is null, even when the schema
//!   declares a default value for it. Message fields, members of a `oneof`, `optional` fields and
//!   all singular proto2 fields have explicit presence.
//! - A field with implicit presence, which is a plain proto3 scalar, is non-nullable. An absent
//!   field holds the protobuf default value. A `double` or `float` field keeps a `-0.0` that the
//!   message holds, and [`encode_messages`] and [`ProtobufValue::encode_to_vec`] write it, although
//!   it equals the default zero.
//! - A `repeated` field is a non-nullable list with non-nullable elements, and an absent field is
//!   an empty list.
//! - A `map<K, V>` field is a non-nullable list of non-nullable `{key, value}` structs with a
//!   non-nullable key and value. The decoder writes the entries in key order, and a missing key or
//!   value holds its default.
//! - A group is stored as a message.
//!
//! Single values map as follows:
//!
//! | Protobuf                          | Vortex                                    |
//! | --------------------------------- | ----------------------------------------- |
//! | `double`                          | `f64`                                     |
//! | `float`                           | `f32`                                     |
//! | `int32`, `sint32`, `sfixed32`     | `i32`                                     |
//! | `int64`, `sint64`, `sfixed64`     | `i64`                                     |
//! | `uint32`, `fixed32`               | `u32`                                     |
//! | `uint64`, `fixed64`               | `u64`                                     |
//! | `bool`                            | `bool`                                    |
//! | `string`                          | `utf8`                                    |
//! | `bytes`                           | `binary`                                  |
//! | enum                              | `i32`, the enum number                    |
//! | message                           | struct                                    |
//! | message beyond the recursion depth | `binary`, the encoding of its fields     |
//! | wrapper, such as `Int32Value`, with [`WrapperMapping::Primitive`] | the wrapped value |
//! | `Timestamp` with [`TimestampMapping::Nanoseconds`] | `vortex.timestamp` in nanoseconds, UTC |
//! | `Struct`, `Value`, `ListValue` with [`JsonMapping::Text`] | `vortex.json`        |
//!
//! An enum keeps its number, so a number that an open (proto3) enum does not define survives.
//! [`protobuf_enum_name`] maps the numbers to value names, with [`EnumName`].
//!
//! ## Options
//!
//! | Option | Default | Effect |
//! | ------ | ------- | ------ |
//! | [`preserve_unknown_fields`](MappingOptions::preserve_unknown_fields) | off | A binary [`UNKNOWN_FIELDS`] field of each message holds the fields that the schema does not know, as read, and encoding writes them back after the known fields. |
//! | [`max_recursion_depth`](MappingOptions::max_recursion_depth) | none | Without it, recursive messages are rejected. With depth `n`, a message occurs as a struct at most `n` times inside itself, and a deeper occurrence is stored as its encoding. |
//! | [`wrappers`](MappingOptions::wrappers) | [`WrapperMapping::Message`] | How to store `google.protobuf` wrapper messages. |
//! | [`timestamps`](MappingOptions::timestamps) | [`TimestampMapping::Message`] | How to store `google.protobuf.Timestamp`. |
//! | [`json`](MappingOptions::json) | [`JsonMapping::Message`] | How to store `google.protobuf.Struct`, `Value` and `ListValue`. |
//!
//! A message type is a well-known type only if it has the full name and the fields of one. The
//! dedicated forms apply to fields, not to the root message. `google.protobuf.Duration` stays a
//! message, because Vortex has no duration type, and so do the other well-known types, such as
//! `Any` and `FieldMask`. The JSON types contain each other, so their default mapping needs a
//! recursion depth.
//!
//! A recursion depth multiplies the fields of a recursive message, and so does a message type that
//! occurs at several places of a schema. The storage of a message can have at most 100,000 fields
//! in all, in both mapping versions. The bytes stored at the depth are not parsed when decoding,
//! like a lazy field: a malformed message beyond the depth decodes, and fails only when it is
//! parsed.
//!
//! ## What a round trip keeps
//!
//! Decoding a message and encoding it again gives a message that parses to the same message, with
//! these exceptions:
//!
//! - Unknown fields and extensions are dropped, and so are values whose wire type does not match
//!   their field and numbers that a closed (proto2) enum does not define, which protobuf parsers
//!   keep as unknown fields. With [`preserve_unknown_fields`](MappingOptions::preserve_unknown_fields)
//!   they are kept, except inside map entries and inside messages stored in a dedicated form.
//! - The encoder writes the unknown fields of a message after its known fields. Protobuf resolves
//!   a field that occurs several times by its order, so when a newer schema adds a member to a
//!   oneof, or a number to a closed enum, a known value and an unknown one can swap which wins.
//! - The dedicated forms of the well-known types drop unknown fields inside those messages. A
//!   value without a dedicated form fails to decode rather than lose information: a timestamp
//!   outside the `i64` nanosecond range or with `nanos` outside `0..=999_999_999`, and a JSON
//!   `Value` without a kind or with a NaN or infinite number. A `null_value` becomes JSON `null`,
//!   so a number that `NullValue` does not define encodes as `NULL_VALUE`, as in the protobuf
//!   JSON mapping.
//! - A proto2 `string` field must hold valid UTF-8, which the `utf8` storage needs.
//!
//! The encoding itself is canonical: fields in field number order, packed repeated fields where
//! the schema declares them packed, and map entries in key order. JSON text has sorted object keys
//! and numbers in the shortest form that reads back as the same double.
//!
//! ## Protobuf editions
//!
//! Schemas in editions syntax, such as `edition = "2023"`, are not supported yet. Editions set
//! field presence, packed encoding, delimited (group) encoding, closed enums and UTF-8 checks with
//! features per file, message or field. `prost-reflect` 0.16 rejects editions descriptors, and
//! `protox` 0.9 cannot parse editions files, so no message descriptor of an editions file exists,
//! and metadata with an editions descriptor set fails to load. Supporting editions also needs the
//! canonical descriptor set to keep the `edition` and the features.
//!
//! # Files
//!
//! A Vortex file can split the storage struct of a protobuf column into one column per message
//! field, so a scan that reads one field does not read whole messages. This needs the experimental
//! extension layout of `vortex-layout`, which [`initialize`] opts protobuf columns into, and a
//! writer whose enabled editions include it, as the [`UNSTABLE`](vortex_edition::UNSTABLE)
//! edition does. A protobuf column with JSON text also needs the JSON edition, which [`initialize`]
//! enables through `vortex_json::initialize`.

mod canonical;
mod decode;
mod dtype;
mod encode;
mod enum_name;
mod field;
mod mapping;
mod metadata;
mod options;
mod pool;
mod value;
mod well_known;
mod wire;

pub use decode::ProtobufDecoder;
pub use decode::decode_messages;
pub use dtype::Protobuf;
pub use encode::encode_messages;
pub use enum_name::EnumName;
pub use enum_name::EnumNameOptions;
pub use field::protobuf_enum_name;
pub use field::protobuf_field;
pub use field::protobuf_oneof_case;
pub use mapping::UNKNOWN_FIELDS;
pub use metadata::MAPPING_VERSION;
pub use metadata::ProtobufMetadata;
pub use options::JsonMapping;
pub use options::MAX_RECURSION_DEPTH;
pub use options::MappingOptions;
pub use options::TimestampMapping;
pub use options::WrapperMapping;
pub use value::ProtobufValue;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_edition::ComponentKind;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::unstable;
use vortex_error::VortexExpect;
use vortex_layout::session::LayoutSessionExt;
use vortex_session::VortexSession;

/// Registers protobuf extension support with a session: the dtype, the [`EnumName`] function and
/// JSON support, which `vortex_json::initialize` registers.
///
/// The dtype joins the experimental [`UNSTABLE`](vortex_edition::UNSTABLE) edition, which has no
/// compatibility guarantee. A writer can only store the dtype in files after it opts in with
/// `session.enable_edition(UNSTABLE)`. If the session has not declared that edition, as in an
/// empty session, this declares it. Protobuf columns are opted into the extension layout, which
/// the same edition includes. Arrow conversions need no plugin: an array exports as its storage
/// struct with the dtype's Vortex ID and metadata, and imports back to the same dtype.
pub fn initialize(session: &VortexSession) {
    vortex_json::initialize(session);
    session.dtypes().register(Protobuf);
    session.scalar_fns().register(EnumName);
    session
        .layouts()
        .decompose_extension(ExtVTable::id(&Protobuf));

    if session
        .editions()
        .find_family(unstable::FAMILY.name)
        .is_none()
    {
        session
            .editions()
            .declare_family(&unstable::FAMILY)
            .vortex_expect("unstable edition family is valid");
    }
    if session.editions().find(&unstable::UNSTABLE).is_none() {
        session
            .register_edition(&unstable::DECLARATION)
            .vortex_expect("unstable edition declaration is valid");
    }

    // `initialize` is idempotent, so only declare the membership once.
    let id = ExtVTable::id(&Protobuf);
    let declared = session
        .editions()
        .components_in(&unstable::UNSTABLE, ComponentKind::DType)
        .iter()
        .any(|inclusion| inclusion.component_id == id);
    if !declared {
        session
            .editions()
            .declare_inclusion(EditionInclusion::dtype(&id, unstable::UNSTABLE))
            .vortex_expect("protobuf dtype joins the unstable edition once");
    }
}

#[cfg(test)]
mod tests;

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]

//! An experimental extension type that stores protobuf messages of one schema in a Vortex column.
//!
//! The [`Protobuf`] dtype, with ID `vortex.protobuf`, keeps the message schema in its
//! [`ProtobufMetadata`] and stores each message as a struct with one field per message field.
//! Query engines read single fields through the storage without decoding whole messages. The
//! [`protobuf_field`] expression reads a field by its dotted path.
//!
//! [`ProtobufDecoder`] builds arrays from serialized messages, and [`encode_messages`] serializes
//! them again. [`initialize`] registers the dtype with a session.
//!
//! # Storage mapping
//!
//! The storage dtype follows from the message descriptor and the mapping version in the metadata.
//! Mapping version 1 is the only version so far:
//!
//! - A message is a struct whose fields have the protobuf field names, in declaration order.
//! - A field with explicit presence is nullable, and an absent field is null, even when the schema
//!   declares a default value for it. Message fields, members of a `oneof`, `optional` fields and
//!   all singular proto2 fields have explicit presence.
//! - A field with implicit presence, which is a plain proto3 scalar, is non-nullable. An absent
//!   field holds the protobuf default value. A `double` or `float` field keeps a `-0.0` that the
//!   message holds. [`encode_messages`] and [`ProtobufValue::encode_to_vec`] do not write it,
//!   because `prost-reflect`, which encodes the messages, treats it as the default zero, so it
//!   reads back as `0.0`.
//! - A `repeated` field is a non-nullable list with non-nullable elements, and an absent field is
//!   an empty list.
//! - A `map<K, V>` field is a non-nullable list of non-nullable `{key, value}` structs with a
//!   non-nullable key and value. The decoder writes the entries in key order.
//!
//! Scalar values map as follows:
//!
//! | Protobuf                          | Vortex   |
//! | --------------------------------- | -------- |
//! | `double`                          | `f64`    |
//! | `float`                           | `f32`    |
//! | `int32`, `sint32`, `sfixed32`     | `i32`    |
//! | `int64`, `sint64`, `sfixed64`     | `i64`    |
//! | `uint32`, `fixed32`               | `u32`    |
//! | `uint64`, `fixed64`               | `u64`    |
//! | `bool`                            | `bool`   |
//! | `string`                          | `utf8`   |
//! | `bytes`                           | `binary` |
//! | enum                              | `i32` (the enum number) |
//!
//! Groups and well-known types such as `google.protobuf.Timestamp` are stored as ordinary
//! messages. Recursive messages, which contain themselves directly or through other messages, have
//! no storage dtype and are rejected. Unknown fields and extensions are dropped on decode.
//!
//! # Files
//!
//! A Vortex file can split the storage struct of a protobuf column into one column per message
//! field, so a scan that reads one field does not read whole messages. This needs the experimental
//! extension layout of `vortex-layout`, which [`initialize`] opts protobuf columns into, and a
//! writer whose enabled editions include it, as the [`UNSTABLE`](vortex_edition::UNSTABLE)
//! edition does.

mod canonical;
mod decode;
mod dtype;
mod encode;
mod field;
mod mapping;
mod metadata;
mod value;

pub use decode::ProtobufDecoder;
pub use decode::decode_messages;
pub use dtype::Protobuf;
pub use encode::encode_messages;
pub use field::protobuf_field;
pub use metadata::MAPPING_VERSION;
pub use metadata::ProtobufMetadata;
pub use value::ProtobufValue;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_edition::ComponentKind;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::unstable;
use vortex_error::VortexExpect;
use vortex_layout::session::LayoutSessionExt;
use vortex_session::VortexSession;

/// Registers protobuf extension support with a session.
///
/// The dtype joins the experimental [`UNSTABLE`](vortex_edition::UNSTABLE) edition, which has no
/// compatibility guarantee. A writer can only store the dtype in files after it opts in with
/// `session.enable_edition(UNSTABLE)`. If the session has not declared that edition, as in an
/// empty session, this declares it. Protobuf columns are opted into the extension layout, which
/// the same edition includes. Arrow conversions need no plugin: an array exports as its storage
/// struct with the dtype's Vortex ID and metadata, and imports back to the same dtype.
pub fn initialize(session: &VortexSession) {
    session.dtypes().register(Protobuf);
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

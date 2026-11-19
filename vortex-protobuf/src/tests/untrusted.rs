// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stored metadata from untrusted files, which fails to load instead of panicking or exhausting
//! memory when its descriptor set is malformed or its schema expands to too many fields.

use prost::Message;
use prost_types::DescriptorProto;
use prost_types::FieldDescriptorProto;
use prost_types::FileDescriptorProto;
use prost_types::FileDescriptorSet;
use prost_types::MessageOptions;
use prost_types::field_descriptor_proto::Label;
use prost_types::field_descriptor_proto::Type;
use rstest::rstest;
use vortex_array::dtype::extension::ExtVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::message;
use crate::MAPPING_VERSION;
use crate::Protobuf;
use crate::canonical::canonical_descriptor_set;
use crate::mapping::MAX_STORAGE_FIELDS;
use crate::metadata::SerializedMetadata;

/// Returns a field named `name` with the given number, label and type, which names a message
/// type for [`Type::Message`].
fn field(name: &str, number: i32, label: Label, field_type: Type) -> FieldDescriptorProto {
    FieldDescriptorProto {
        name: Some(name.to_owned()),
        number: Some(number),
        label: Some(label.into()),
        r#type: Some(field_type.into()),
        type_name: (field_type == Type::Message).then(|| ".crafted.Other".to_owned()),
        ..Default::default()
    }
}

/// Serializes version 2 metadata of `crafted.Root`, whose field `entries` is a map with an entry
/// message of the given fields.
fn map_entry_metadata(entry_fields: Vec<FieldDescriptorProto>) -> Vec<u8> {
    let entry = DescriptorProto {
        name: Some("EntriesEntry".to_owned()),
        field: entry_fields,
        options: Some(MessageOptions {
            map_entry: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut entries = field("entries", 1, Label::Repeated, Type::Message);
    entries.type_name = Some(".crafted.Root.EntriesEntry".to_owned());
    let root = DescriptorProto {
        name: Some("Root".to_owned()),
        field: vec![entries],
        nested_type: vec![entry],
        ..Default::default()
    };
    let other = DescriptorProto {
        name: Some("Other".to_owned()),
        ..Default::default()
    };
    let descriptor_set = FileDescriptorSet {
        file: vec![FileDescriptorProto {
            name: Some("crafted.proto".to_owned()),
            package: Some("crafted".to_owned()),
            syntax: Some("proto3".to_owned()),
            message_type: vec![root, other],
            ..Default::default()
        }],
    };
    SerializedMetadata {
        mapping_version: MAPPING_VERSION,
        message_name: "crafted.Root".to_owned(),
        descriptor_set: descriptor_set.encode_to_vec(),
        options: None,
    }
    .encode_to_vec()
}

/// A map entry message must have the fields that `protoc` generates for a map: a singular key
/// field 1 of an integer, bool or string type and a singular value field 2. `prost-reflect`
/// accepts other entry messages, and panics when it looks up a key or value field that is missing.
#[rstest]
#[case::valid(
    vec![
        field("key", 1, Label::Optional, Type::String),
        field("value", 2, Label::Optional, Type::Message),
    ],
    None
)]
#[case::missing_key(
    vec![field("value", 2, Label::Optional, Type::Int32)],
    Some("must have a singular key field 1 and a singular value field 2")
)]
#[case::missing_value(
    vec![field("key", 1, Label::Optional, Type::String)],
    Some("must have a singular key field 1 and a singular value field 2")
)]
#[case::repeated_key(
    vec![
        field("key", 1, Label::Repeated, Type::String),
        field("value", 2, Label::Optional, Type::Int32),
    ],
    Some("must have a singular key field 1 and a singular value field 2")
)]
#[case::repeated_value(
    vec![
        field("key", 1, Label::Optional, Type::String),
        field("value", 2, Label::Repeated, Type::Int32),
    ],
    Some("must have a singular key field 1 and a singular value field 2")
)]
#[case::extra_field(
    vec![
        field("key", 1, Label::Optional, Type::String),
        field("value", 2, Label::Optional, Type::Int32),
        field("extra", 3, Label::Optional, Type::Int32),
    ],
    Some("must have a singular key field 1 and a singular value field 2")
)]
#[case::message_key(
    vec![
        field("key", 1, Label::Optional, Type::Message),
        field("value", 2, Label::Optional, Type::Int32),
    ],
    Some("has a key that is not an integer, bool or string")
)]
#[case::double_key(
    vec![
        field("key", 1, Label::Optional, Type::Double),
        field("value", 2, Label::Optional, Type::Int32),
    ],
    Some("has a key that is not an integer, bool or string")
)]
#[case::bytes_key(
    vec![
        field("key", 1, Label::Optional, Type::Bytes),
        field("value", 2, Label::Optional, Type::Int32),
    ],
    Some("has a key that is not an integer, bool or string")
)]
fn test_malformed_map_entry_is_rejected(
    #[case] entry_fields: Vec<FieldDescriptorProto>,
    #[case] expected: Option<&str>,
) -> VortexResult<()> {
    let metadata = map_entry_metadata(entry_fields);
    match (Protobuf.deserialize_metadata(&metadata), expected) {
        (Ok(_), None) => {}
        (Err(error), Some(expected)) => {
            let error = error.to_string();
            assert!(
                error.contains("Invalid protobuf descriptor set: map entry message")
                    && error.contains("crafted.Root.EntriesEntry")
                    && error.contains(expected),
                "{error}"
            );
        }
        (result, expected) => {
            return Err(vortex_err!("got {result:?}, expected {expected:?}"));
        }
    }
    Ok(())
}

/// Returns a schema whose message `wide.L{depth}` holds two fields of type `L{depth - 1}`, down
/// to `L0`, which holds one field. The schema is not recursive, but the storage of `L{depth}` has
/// `3 * 2^depth - 2` fields.
fn doubling_schema(depth: usize) -> String {
    let mut source =
        String::from("syntax = \"proto3\";\npackage wide;\nmessage L0 { int32 a = 1; }\n");
    for level in 1..=depth {
        let inner = level - 1;
        source.push_str(&format!(
            "message L{level} {{ L{inner} x = 1; L{inner} y = 2; }}\n"
        ));
    }
    source
}

/// The storage of a message has at most [`MAX_STORAGE_FIELDS`] fields in every mapping version.
/// Message types that occur at several places expand at each, so a small schema without recursion
/// can need more fields than memory holds, and its metadata fails to load quickly instead.
#[rstest]
#[case::within_limit(15, true)]
#[case::over_limit(16, false)]
#[case::far_over_limit(60, false)]
fn test_storage_field_limit_applies_to_every_version(
    #[case] depth: usize,
    #[case] loads: bool,
    #[values(1, MAPPING_VERSION)] mapping_version: u32,
) -> VortexResult<()> {
    let source = doubling_schema(depth);
    let name = format!("wide.L{depth}");
    let descriptor = message(&[("wide.proto", source.as_str())], &name)?;
    let metadata = SerializedMetadata {
        mapping_version,
        message_name: name,
        descriptor_set: canonical_descriptor_set(&descriptor).encode_to_vec(),
        options: None,
    }
    .encode_to_vec();
    match Protobuf.deserialize_metadata(&metadata) {
        Ok(_) => assert!(loads, "a schema of {depth} levels loaded"),
        Err(error) => {
            assert!(!loads, "{error}");
            assert!(
                error
                    .to_string()
                    .contains(&format!("needs more than {MAX_STORAGE_FIELDS} fields")),
                "{error}"
            );
        }
    }
    Ok(())
}

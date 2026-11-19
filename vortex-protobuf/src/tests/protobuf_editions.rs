// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Schemas in protobuf editions syntax, such as `edition = "2023"`.
//!
//! An editions file sets field presence, packed encoding, delimited (group) encoding, closed
//! enums and UTF-8 checks per file, message and field with features, where proto2 and proto3 fix
//! them per syntax. The protobuf libraries of this crate do not support editions yet: `protox`
//! 0.9 cannot parse an editions file, and `prost-reflect` 0.16, the latest release, rejects a
//! descriptor with `syntax = "editions"`. `prost-types` 0.14 has no `edition` field or features.
//! So no message descriptor of an editions file exists, and stored metadata with an editions
//! descriptor set fails to load instead of being misread. The canonical descriptor set keeps
//! `syntax` but drops `edition` and the features, which supporting editions must change.

use prost::Message;
use prost_types::DescriptorProto;
use prost_types::FieldDescriptorProto;
use prost_types::FileDescriptorProto;
use prost_types::FileDescriptorSet;
use prost_types::field_descriptor_proto::Label;
use prost_types::field_descriptor_proto::Type;
use vortex_array::dtype::extension::ExtVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::compile;
use crate::MAPPING_VERSION;
use crate::Protobuf;
use crate::metadata::SerializedMetadata;

/// An editions file with a field of each feature that differs from proto3.
const EDITIONS_PROTO: (&str, &str) = (
    "editions.proto",
    r#"
edition = "2023";
package editions;

message Event {
  int32 explicit = 1;
  int32 implicit = 2 [features.field_presence = IMPLICIT];
  repeated int32 expanded = 3 [features.repeated_field_encoding = EXPANDED];
  Event delimited = 4 [features.message_encoding = DELIMITED];
}
"#,
);

/// `protox` cannot compile an editions file.
#[test]
fn test_protox_rejects_editions() -> VortexResult<()> {
    let error = compile(&[EDITIONS_PROTO])
        .err()
        .ok_or_else(|| vortex_err!("protox compiled an editions file"))?;
    assert!(error.to_string().contains("Invalid test schema"), "{error}");
    Ok(())
}

/// Metadata whose descriptor set holds an editions file fails to load.
#[test]
fn test_editions_descriptor_set_is_rejected() -> VortexResult<()> {
    let descriptor_set = FileDescriptorSet {
        file: vec![FileDescriptorProto {
            name: Some("editions.proto".to_owned()),
            package: Some("editions".to_owned()),
            syntax: Some("editions".to_owned()),
            message_type: vec![DescriptorProto {
                name: Some("Event".to_owned()),
                field: vec![FieldDescriptorProto {
                    name: Some("explicit".to_owned()),
                    number: Some(1),
                    label: Some(Label::Optional.into()),
                    r#type: Some(Type::Int32.into()),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }],
    };
    let metadata = SerializedMetadata {
        mapping_version: MAPPING_VERSION,
        message_name: "editions.Event".to_owned(),
        descriptor_set: descriptor_set.encode_to_vec(),
        options: None,
    }
    .encode_to_vec();

    let error = Protobuf
        .deserialize_metadata(&metadata)
        .err()
        .ok_or_else(|| vortex_err!("an editions descriptor set was accepted"))?;
    let message = error.to_string();
    assert!(
        message.contains("Invalid protobuf descriptor set") && message.contains("editions"),
        "{message}"
    );
    Ok(())
}

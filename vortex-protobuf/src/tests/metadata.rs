// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Metadata serialization, canonicalization and validation.

use prost::Message;
use prost_reflect::DescriptorPool;
use prost_types::DescriptorProto;
use prost_types::FileDescriptorSet;
use rstest::rstest;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtDTypePlugin;
use vortex_array::dtype::extension::ExtVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::RECURSIVE_PROTO;
use super::TEST_PROTO;
use super::UNRELATED_PROTO;
use super::all_types_dtype;
use super::compile;
use super::message;
use super::protobuf_dtype;
use crate::MAPPING_VERSION;
use crate::Protobuf;
use crate::ProtobufMetadata;
use crate::metadata::SerializedMetadata;

/// The schema of `shop.Order`, which the cases of [`test_metadata_ignores_unrelated_changes`] and
/// [`test_metadata_differs_by_reachable_types`] change.
const SHOP_PROTO: &str = r#"syntax = "proto3";
package shop;

message Address {
  string city = 1;
}

message Order {
  int64 id = 1;
  Address address = 2;
  repeated int64 tags = 3;
}
"#;

/// Returns [`SHOP_PROTO`] with `addition` right after the package, where imports can go too.
fn shop_with(addition: &str) -> String {
    SHOP_PROTO.replacen(
        "package shop;\n",
        &format!("package shop;\n\n{addition}\n"),
        1,
    )
}

/// Returns [`SHOP_PROTO`] with `from` replaced by `to`.
fn shop_replacing(from: &str, to: &str) -> String {
    assert!(SHOP_PROTO.contains(from), "{from}");
    SHOP_PROTO.replacen(from, to, 1)
}

/// A schema whose message `Holder.Leaf` uses an enum nested in the message that encloses it.
const NESTED_PROTO: (&str, &str) = (
    "nested.proto",
    r#"
syntax = "proto3";
package nested;

import "google/protobuf/timestamp.proto";

message Holder {
  enum Kind {
    KIND_UNSPECIFIED = 0;
    KIND_LEAF = 1;
  }
  message Leaf {
    Kind kind = 1;
  }
  google.protobuf.Timestamp created = 1;
  Leaf leaf = 2;
}
"#,
);

/// Returns the descriptor set of `pool`, with its files in reverse order.
fn reversed_descriptor_set(pool: &DescriptorPool) -> FileDescriptorSet {
    let mut file = pool.file_descriptor_protos().cloned().collect::<Vec<_>>();
    file.reverse();
    FileDescriptorSet { file }
}

/// Returns the metadata of the message `name` compiled from `sources`.
fn metadata(sources: &[(&str, &str)], name: &str) -> VortexResult<ProtobufMetadata> {
    ProtobufMetadata::try_new(&message(sources, name)?)
}

/// Lists the files of a serialized descriptor set with their imports, and the types they define
/// with the fields of each message.
fn describe_descriptor_set(bytes: &[u8]) -> VortexResult<String> {
    let mut lines = Vec::new();
    for file in FileDescriptorSet::decode(bytes)?.file {
        assert!(file.source_code_info.is_none() && file.options.is_none());
        assert!(file.service.is_empty() && file.extension.is_empty());
        lines.push(format!("{} imports {:?}", file.name(), file.dependency));
        for enum_type in &file.enum_type {
            lines.push(format!("  enum {}", enum_type.name()));
        }
        describe_messages(&file.message_type, 1, &mut lines);
    }
    Ok(lines.join("\n"))
}

/// Appends the lines of `messages` for [`describe_descriptor_set`], indented by `depth` levels.
fn describe_messages(messages: &[DescriptorProto], depth: usize, lines: &mut Vec<String>) {
    let indent = "  ".repeat(depth);
    for message in messages {
        let fields = message
            .field
            .iter()
            .map(|field| field.name())
            .collect::<Vec<_>>();
        lines.push(format!("{indent}message {} {fields:?}", message.name()));
        for enum_type in &message.enum_type {
            lines.push(format!("{indent}  enum {}", enum_type.name()));
        }
        describe_messages(&message.nested_type, depth + 1, lines);
    }
}

#[test]
fn test_metadata_serialization_round_trip() -> VortexResult<()> {
    let dtype = all_types_dtype(Nullability::Nullable)?;
    let bytes = Protobuf.serialize_metadata(dtype.metadata())?;
    let decoded = Protobuf.deserialize_metadata(&bytes)?;

    assert_eq!(&decoded, dtype.metadata());
    assert_eq!(decoded.mapping_version(), MAPPING_VERSION);
    assert_eq!(decoded.message_name(), "test.AllTypes");
    assert_eq!(Protobuf.serialize_metadata(&decoded)?, bytes);

    let restored = Protobuf.deserialize(&bytes, dtype.storage_dtype().clone())?;
    assert_eq!(restored, dtype.erased());
    Ok(())
}

/// The descriptor set keeps only the types reachable from the root message and the files that
/// define them, and the root message's dtype builds from it.
#[rstest]
#[case::all_types(&[TEST_PROTO, UNRELATED_PROTO], "test.AllTypes", r#"
google/protobuf/timestamp.proto imports []
  message Timestamp ["seconds", "nanos"]
test.proto imports ["google/protobuf/timestamp.proto"]
  enum Color
  message AllTypes ["f_double", "f_float", "f_int32", "f_int64", "f_uint32", "f_uint64", "f_sint32", "f_sint64", "f_fixed32", "f_fixed64", "f_sfixed32", "f_sfixed64", "f_bool", "f_string", "f_bytes", "f_enum", "inner", "outer", "r_int32", "r_string", "r_inner", "m_inner", "m_labels", "c_text", "c_inner", "o_int32", "o_string", "created"]
    message MInnerEntry ["key", "value"]
    message MLabelsEntry ["key", "value"]
  message Inner ["name", "count"]
  message Outer ["inner", "items"]
"#)]
#[case::other_root(&[TEST_PROTO], "test.Inner", r#"
test.proto imports []
  message Inner ["name", "count"]
"#)]
#[case::nested_message(&[NESTED_PROTO], "nested.Holder.Leaf", r#"
nested.proto imports []
  message Holder []
    enum Kind
    message Leaf ["kind"]
"#)]
fn test_metadata_keeps_only_reachable_types(
    #[case] sources: &[(&str, &str)],
    #[case] name: &str,
    #[case] expected: &str,
) -> VortexResult<()> {
    let dtype = protobuf_dtype(sources, name, Nullability::NonNullable)?;
    assert_eq!(
        describe_descriptor_set(dtype.metadata().descriptor_set())?,
        expected.trim()
    );
    assert_eq!(dtype.metadata().message_name(), name);
    Ok(())
}

#[test]
fn test_metadata_is_canonical() -> VortexResult<()> {
    let pool = compile(&[UNRELATED_PROTO, TEST_PROTO])?;
    let reversed = DescriptorPool::from_file_descriptor_set(reversed_descriptor_set(&pool))
        .map_err(|err| vortex_err!("{err}"))?;
    let name = "test.AllTypes";

    let metadata = |pool: &DescriptorPool| {
        pool.get_message_by_name(name)
            .ok_or_else(|| vortex_err!("missing {name}"))
            .and_then(|descriptor| ProtobufMetadata::try_new(&descriptor))
    };
    let original = metadata(&pool)?;
    let reordered = metadata(&reversed)?;

    assert_eq!(original, reordered);
    assert_eq!(original.descriptor_set(), reordered.descriptor_set());
    assert_eq!(
        Protobuf.serialize_metadata(&original)?,
        Protobuf.serialize_metadata(&reordered)?
    );
    Ok(())
}

/// Changes to a file that leave the types reachable from the root message as they are do not
/// change the metadata: other types, services, extensions, unused imports, options that do not
/// change the wire format and the order of declarations.
#[rstest]
#[case::other_message(shop_with("message Refund {\n  string reason = 1;\n}"))]
#[case::other_enum(shop_with("enum Status {\n  STATUS_UNSPECIFIED = 0;\n}"))]
#[case::service(shop_with("service Shop {\n  rpc Get(Address) returns (Order);\n}"))]
#[case::unused_import(shop_with("import \"google/protobuf/duration.proto\";"))]
#[case::custom_option(
    shop_with(
        "import \"google/protobuf/descriptor.proto\";\n\
         extend google.protobuf.FieldOptions {\n  string label = 50000;\n}",
    )
    .replacen("int64 id = 1;", "int64 id = 1 [(label) = \"key\"];", 1)
)]
#[case::file_option(shop_with("option java_package = \"com.example.shop\";"))]
#[case::deprecated_field(shop_replacing("int64 id = 1;", "int64 id = 1 [deprecated = true];"))]
#[case::json_name(shop_replacing("int64 id = 1;", "int64 id = 1 [json_name = \"orderId\"];"))]
#[case::declaration_order(
    shop_replacing("message Address {\n  string city = 1;\n}\n", "")
        + "\nmessage Address {\n  string city = 1;\n}\n"
)]
fn test_metadata_ignores_unrelated_changes(#[case] source: String) -> VortexResult<()> {
    let base = metadata(&[("shop.proto", SHOP_PROTO)], "shop.Order")?;
    let changed = metadata(&[("shop.proto", &source)], "shop.Order")?;
    assert_eq!(changed, base);
    Ok(())
}

/// Changes to a reachable type change the metadata, as does choosing another root message.
#[rstest]
#[case::field_number(shop_replacing("string city = 1;", "string city = 4;"))]
#[case::field_type(shop_replacing("string city = 1;", "bytes city = 1;"))]
#[case::packed_option(shop_replacing(
    "repeated int64 tags = 3;",
    "repeated int64 tags = 3 [packed = false];"
))]
fn test_metadata_differs_by_reachable_types(#[case] source: String) -> VortexResult<()> {
    let base = metadata(&[("shop.proto", SHOP_PROTO)], "shop.Order")?;
    let changed = metadata(&[("shop.proto", &source)], "shop.Order")?;
    assert_ne!(changed.descriptor_set(), base.descriptor_set());

    let address = metadata(&[("shop.proto", SHOP_PROTO)], "shop.Address")?;
    assert_ne!(address.descriptor_set(), base.descriptor_set());
    Ok(())
}

/// The debug output summarizes the descriptor set instead of printing its bytes.
#[test]
fn test_metadata_debug_summarizes_descriptor_set() -> VortexResult<()> {
    let metadata = all_types_dtype(Nullability::NonNullable)?
        .metadata()
        .clone();
    let debug = format!("{metadata:?}");
    let expected = format!(
        "ProtobufMetadata {{ mapping_version: 1, message_name: \"test.AllTypes\", \
         descriptor_set: <{} bytes, hash ",
        metadata.descriptor_set().len()
    );
    assert!(debug.starts_with(&expected), "{debug}");
    Ok(())
}

#[rstest]
#[case::direct("recursive.Node", "recursive.Node -> recursive.Node")]
#[case::mutual("recursive.Ping", "recursive.Ping -> recursive.Pong -> recursive.Ping")]
#[case::map_value(
    "recursive.Tree",
    "recursive.Tree -> recursive.Tree.BranchesEntry -> recursive.Tree"
)]
#[case::nested(
    "recursive.Holder",
    "recursive.Ping -> recursive.Pong -> recursive.Ping"
)]
fn test_validate_rejects_recursive_messages(
    #[case] name: &str,
    #[case] cycle: &str,
) -> VortexResult<()> {
    let metadata = ProtobufMetadata::try_new(&message(&[RECURSIVE_PROTO], name)?)?;
    let error = ExtDType::<Protobuf>::try_new(metadata, Nullability::NonNullable)
        .err()
        .ok_or_else(|| vortex_err!("{name} was accepted"))?;

    assert!(
        error.to_string().contains(&format!(
            "Recursive protobuf messages are not supported: {cycle}"
        )),
        "{error}"
    );
    Ok(())
}

#[rstest]
#[case::unknown_mapping_version(
    2,
    "test.AllTypes",
    "Unsupported protobuf storage mapping version 2"
)]
#[case::missing_root_message(
    MAPPING_VERSION,
    "test.Missing",
    "does not define the root message \"test.Missing\""
)]
fn test_deserialize_rejects_invalid_metadata(
    #[case] mapping_version: u32,
    #[case] message_name: &str,
    #[case] expected: &str,
) -> VortexResult<()> {
    let dtype = all_types_dtype(Nullability::NonNullable)?;
    let bytes = SerializedMetadata {
        mapping_version,
        message_name: message_name.to_owned(),
        descriptor_set: dtype.metadata().descriptor_set().to_vec(),
    }
    .encode_to_vec();

    let error = Protobuf
        .deserialize(&bytes, dtype.storage_dtype().clone())
        .err()
        .ok_or_else(|| vortex_err!("invalid metadata was accepted"))?;
    assert!(error.to_string().contains(expected), "{error}");
    Ok(())
}

#[test]
fn test_deserialize_rejects_mismatched_storage() -> VortexResult<()> {
    let dtype = all_types_dtype(Nullability::NonNullable)?;
    let bytes = Protobuf.serialize_metadata(dtype.metadata())?;
    let inner = protobuf_dtype(&[TEST_PROTO], "test.Inner", Nullability::NonNullable)?;

    let error = Protobuf
        .deserialize(&bytes, inner.storage_dtype().clone())
        .err()
        .ok_or_else(|| vortex_err!("mismatched storage was accepted"))?;
    assert!(
        error
            .to_string()
            .contains("vortex.protobuf storage must be {f_double=f64, "),
        "{error}"
    );
    Ok(())
}

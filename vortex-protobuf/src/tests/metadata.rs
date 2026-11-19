// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Metadata serialization, canonicalization and validation.

use std::sync::Arc;

use prost::Message;
use prost_reflect::DescriptorPool;
use prost_types::DescriptorProto;
use prost_types::FileDescriptorProto;
use prost_types::FileDescriptorSet;
use rstest::rstest;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDTypePlugin;
use vortex_array::dtype::extension::ExtVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::LEGACY_PROTO;
use super::RECURSIVE_PROTO;
use super::TEST_PROTO;
use super::UNRELATED_PROTO;
use super::all_types_dtype;
use super::compile;
use super::message;
use super::protobuf_dtype;
use crate::JsonMapping;
use crate::MAPPING_VERSION;
use crate::MAX_RECURSION_DEPTH;
use crate::MappingOptions;
use crate::Protobuf;
use crate::ProtobufMetadata;
use crate::TimestampMapping;
use crate::WrapperMapping;
use crate::metadata::SerializedMetadata;
use crate::options::SerializedOptions;
use crate::pool::CAPACITY;
use crate::pool::PoolCache;

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

/// The options survive serialization, and the default options are left out of the bytes.
#[rstest]
#[case::defaults(MappingOptions::default(), false)]
#[case::all_options(
    MappingOptions {
        preserve_unknown_fields: true,
        max_recursion_depth: Some(3),
        wrappers: WrapperMapping::Primitive,
        timestamps: TimestampMapping::Nanoseconds,
        json: JsonMapping::Text,
    },
    true
)]
fn test_metadata_options_round_trip(
    #[case] options: MappingOptions,
    #[case] stored: bool,
) -> VortexResult<()> {
    let descriptor = message(&[TEST_PROTO, UNRELATED_PROTO], "test.AllTypes")?;
    let metadata = ProtobufMetadata::try_new_with_options(&descriptor, options.clone())?;
    let bytes = Protobuf.serialize_metadata(&metadata)?;
    assert_eq!(
        SerializedMetadata::decode(bytes.as_slice())?
            .options
            .is_some(),
        stored
    );

    let decoded = Protobuf.deserialize_metadata(&bytes)?;
    assert_eq!(decoded, metadata);
    assert_eq!(decoded.options(), &options);
    assert_eq!(
        decoded == ProtobufMetadata::try_new(&descriptor)?,
        options == MappingOptions::default()
    );
    Ok(())
}

/// Version 1 metadata, written before the mapping had options, loads with the storage dtype that
/// version 1 derives, which version 2 derives with the default options. It serializes back to the
/// same bytes, so files keep their metadata.
#[rstest]
#[case::proto3(&[TEST_PROTO, UNRELATED_PROTO], "test.AllTypes")]
#[case::proto2(&[LEGACY_PROTO], "legacy.Legacy")]
fn test_version_1_metadata_loads(
    #[case] sources: &[(&str, &str)],
    #[case] name: &str,
) -> VortexResult<()> {
    let current = metadata(sources, name)?;
    let bytes = SerializedMetadata {
        mapping_version: 1,
        message_name: name.to_owned(),
        descriptor_set: current.descriptor_set().to_vec(),
        options: None,
    }
    .encode_to_vec();
    let version_1 = Protobuf.deserialize_metadata(&bytes)?;

    assert_eq!(version_1.mapping_version(), 1);
    assert_eq!(version_1.options(), &MappingOptions::default());
    assert_ne!(version_1, current);
    assert_eq!(Protobuf.serialize_metadata(&version_1)?, bytes);
    let storage = Protobuf::storage_dtype(&current);
    assert_eq!(Protobuf::storage_dtype(&version_1), storage);
    Protobuf.deserialize(&bytes, storage)?;
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
        "ProtobufMetadata {{ mapping_version: 2, options: {:?}, message_name: \
         \"test.AllTypes\", descriptor_set: <{} bytes, hash ",
        MappingOptions::default(),
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
fn test_metadata_rejects_recursive_messages(
    #[case] name: &str,
    #[case] cycle: &str,
) -> VortexResult<()> {
    let error = ProtobufMetadata::try_new(&message(&[RECURSIVE_PROTO], name)?)
        .err()
        .ok_or_else(|| vortex_err!("{name} was accepted"))?;

    assert!(
        error.to_string().contains(&format!(
            "Recursive protobuf messages need the max_recursion_depth mapping option: {cycle}"
        )),
        "{error}"
    );
    Ok(())
}

#[rstest]
#[case::unknown_mapping_version(
    3,
    "test.AllTypes",
    None,
    "Unsupported protobuf storage mapping version 3, expected 1 to 2"
)]
#[case::missing_mapping_version(
    0,
    "test.AllTypes",
    None,
    "Unsupported protobuf storage mapping version 0"
)]
#[case::missing_root_message(
    MAPPING_VERSION,
    "test.Missing",
    None,
    "does not define the root message \"test.Missing\""
)]
#[case::options_in_version_1(
    1,
    "test.AllTypes",
    Some(MappingOptions::default().serialize()),
    "Protobuf storage mapping version 1 has no options"
)]
#[case::unknown_option_value(
    MAPPING_VERSION,
    "test.AllTypes",
    Some(SerializedOptions { timestamps: 7, ..MappingOptions::default().serialize() }),
    "Unknown protobuf timestamp mapping 7"
)]
#[case::recursion_depth_out_of_range(
    MAPPING_VERSION,
    "test.AllTypes",
    Some(SerializedOptions {
        max_recursion_depth: Some(MAX_RECURSION_DEPTH + 1),
        ..MappingOptions::default().serialize()
    }),
    "max_recursion_depth 101 exceeds the limit of 100"
)]
fn test_deserialize_rejects_invalid_metadata(
    #[case] mapping_version: u32,
    #[case] message_name: &str,
    #[case] options: Option<SerializedOptions>,
    #[case] expected: &str,
) -> VortexResult<()> {
    let dtype = all_types_dtype(Nullability::NonNullable)?;
    let bytes = SerializedMetadata {
        mapping_version,
        message_name: message_name.to_owned(),
        descriptor_set: dtype.metadata().descriptor_set().to_vec(),
        options,
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

/// Metadata of one schema share the cached pool and descriptor set bytes, however the schema was
/// compiled and whether the metadata was created or deserialized.
#[test]
fn test_metadata_shares_cached_pool() -> VortexResult<()> {
    let created = metadata(&[TEST_PROTO, UNRELATED_PROTO], "test.AllTypes")?;
    let recompiled = metadata(&[TEST_PROTO], "test.AllTypes")?;
    let deserialized = Protobuf.deserialize_metadata(&Protobuf.serialize_metadata(&created)?)?;

    for other in [&recompiled, &deserialized] {
        assert_eq!(other.message_descriptor(), created.message_descriptor());
        assert_eq!(
            other.descriptor_set().as_ptr(),
            created.descriptor_set().as_ptr()
        );
    }
    Ok(())
}

/// The pool cache keeps a bounded number of descriptor sets and drops the least recently used.
#[test]
fn test_pool_cache_drops_least_recently_used() -> VortexResult<()> {
    let descriptor_set = |index: usize| -> VortexResult<(Vec<u8>, DescriptorPool)> {
        let set = FileDescriptorSet {
            file: vec![FileDescriptorProto {
                name: Some(format!("file{index}.proto")),
                ..Default::default()
            }],
        };
        let bytes = set.encode_to_vec();
        let pool = DescriptorPool::decode(bytes.as_slice())
            .map_err(|err| vortex_err!("invalid descriptor set: {err}"))?;
        Ok((bytes, pool))
    };

    let mut cache = PoolCache::default();
    let sets = (0..=CAPACITY)
        .map(descriptor_set)
        .collect::<VortexResult<Vec<_>>>()?;
    for (bytes, pool) in &sets[..CAPACITY] {
        cache.insert(Arc::from(bytes.as_slice()), pool.clone());
    }
    // The first set is used again, so the second one is the least recently used.
    let (first, _) = &sets[0];
    assert!(cache.get(first).is_some());
    let (last, last_pool) = &sets[CAPACITY];
    let (_, inserted) = cache.insert(Arc::from(last.as_slice()), last_pool.clone());
    assert_eq!(&inserted, last_pool);

    assert!(cache.get(first).is_some());
    assert!(cache.get(&sets[1].0).is_none());
    assert!(
        sets[2..]
            .iter()
            .all(|(bytes, _)| cache.get(bytes).is_some())
    );
    Ok(())
}

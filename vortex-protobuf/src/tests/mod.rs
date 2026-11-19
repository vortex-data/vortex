// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared schemas and helpers for the crate's tests.

mod arrow;
mod codec;
mod differential;
mod editions;
mod enums_and_oneofs;
mod file;
mod generate;
mod malformed;
mod mapping;
mod metadata;
mod oracle;
mod protobuf_editions;
mod recursion;
mod semantics;
mod unknown;
mod untrusted;
mod well_known;

use std::sync::LazyLock;

use prost::encoding::WireType;
use prost::encoding::encode_key;
use prost::encoding::encode_varint;
use prost_reflect::DescriptorPool;
use prost_reflect::DynamicMessage;
use prost_reflect::MessageDescriptor;
use prost_reflect::Value;
use protox::Compiler;
use protox::file::ChainFileResolver;
use protox::file::File;
use protox::file::FileResolver;
use protox::file::GoogleFileResolver;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::MappingOptions;
use crate::Protobuf;
use crate::ProtobufDecoder;
use crate::ProtobufMetadata;
use crate::encode_messages;

/// A session with protobuf support.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

/// A proto3 schema with every scalar kind, nested and repeated messages, maps, a oneof, proto3
/// `optional` fields and a well-known type.
const TEST_PROTO: (&str, &str) = (
    "test.proto",
    r#"
syntax = "proto3";
package test;

import "google/protobuf/timestamp.proto";

enum Color {
  COLOR_UNSPECIFIED = 0;
  COLOR_RED = 1;
}

message Inner {
  string name = 1;
  int64 count = 2;
}

message Outer {
  Inner inner = 1;
  repeated Inner items = 2;
}

message AllTypes {
  double f_double = 1;
  float f_float = 2;
  int32 f_int32 = 3;
  int64 f_int64 = 4;
  uint32 f_uint32 = 5;
  uint64 f_uint64 = 6;
  sint32 f_sint32 = 7;
  sint64 f_sint64 = 8;
  fixed32 f_fixed32 = 9;
  fixed64 f_fixed64 = 10;
  sfixed32 f_sfixed32 = 11;
  sfixed64 f_sfixed64 = 12;
  bool f_bool = 13;
  string f_string = 14;
  bytes f_bytes = 15;
  Color f_enum = 16;
  Inner inner = 17;
  Outer outer = 18;
  repeated int32 r_int32 = 19;
  repeated string r_string = 20;
  repeated Inner r_inner = 21;
  map<string, Inner> m_inner = 22;
  map<int32, string> m_labels = 23;
  oneof choice {
    string c_text = 24;
    Inner c_inner = 25;
  }
  optional int32 o_int32 = 26;
  optional string o_string = 27;
  google.protobuf.Timestamp created = 28;
}
"#,
);

/// A file in the same pool that the test schema does not import.
const UNRELATED_PROTO: (&str, &str) = (
    "unrelated.proto",
    r#"
syntax = "proto3";
package unrelated;

message Unrelated {
  string text = 1;
}
"#,
);

/// A proto2 schema with explicit presence, a custom default and a group.
const LEGACY_PROTO: (&str, &str) = (
    "legacy.proto",
    r#"
syntax = "proto2";
package legacy;

message Legacy {
  optional int32 a = 1;
  required string b = 2;
  optional int64 c = 3 [default = 7];
  repeated uint32 d = 4;
  optional group Settings = 5 {
    optional bool flag = 6;
  }
}
"#,
);

/// A proto3 schema with repeated fields of every scalar kind, one of them not packed, maps with
/// every kind of key, and a field number that is too large for a dense lookup table.
const REPEATED_PROTO: (&str, &str) = (
    "repeated.proto",
    r#"
syntax = "proto3";
package repeated;

enum Level {
  LEVEL_UNSPECIFIED = 0;
  LEVEL_LOW = 1;
  LEVEL_HIGH = 2;
}

message Point {
  sint64 x = 1;
  double y = 2;
  repeated string tags = 3;
}

message Repeated {
  repeated double r_double = 1;
  repeated float r_float = 2;
  repeated int64 r_int64 = 3;
  repeated uint32 r_uint32 = 4;
  repeated uint64 r_uint64 = 5;
  repeated sint32 r_sint32 = 6;
  repeated sint64 r_sint64 = 7;
  repeated fixed32 r_fixed32 = 8;
  repeated fixed64 r_fixed64 = 9;
  repeated sfixed32 r_sfixed32 = 10;
  repeated sfixed64 r_sfixed64 = 11;
  repeated bool r_bool = 12;
  repeated Level r_enum = 13;
  repeated int32 r_unpacked = 14 [packed = false];
  repeated bytes r_bytes = 15;
  repeated Point r_points = 16;
  map<int64, Point> m_points = 17;
  map<bool, bytes> m_flags = 18;
  map<uint64, Level> m_levels = 19;
  map<sfixed32, float> m_weights = 20;
  map<fixed64, string> m_names = 21;
  map<sint64, uint32> m_counts = 22;
  optional double o_double = 23;
  optional float o_float = 24;
  Point far = 5000;
}
"#,
);

/// A proto2 schema with a closed enum in a singular, repeated, packed, map and oneof field, a
/// repeated group and packed repeated fields.
const CLOSED_PROTO: (&str, &str) = (
    "closed.proto",
    r#"
syntax = "proto2";
package closed;

enum Mode {
  MODE_A = 1;
  MODE_B = 2;
  MODE_C = 5;
}

message Closed {
  optional Mode mode = 1;
  repeated Mode modes = 2;
  repeated Mode packed_modes = 3 [packed = true];
  map<string, Mode> named = 4;
  repeated group Entry = 5 {
    optional string name = 6;
    repeated int32 values = 7;
  }
  optional int32 count = 8 [default = 3];
  oneof choice {
    Mode choice_mode = 9;
    string choice_text = 10;
  }
  repeated sint64 packed_ints = 11 [packed = true];
  optional double ratio = 12;
}
"#,
);

/// Schemas with messages that contain themselves.
const RECURSIVE_PROTO: (&str, &str) = (
    "recursive.proto",
    r#"
syntax = "proto3";
package recursive;

message Node {
  repeated Node children = 1;
}

message Ping {
  Pong pong = 1;
}

message Pong {
  Ping ping = 1;
}

message Tree {
  map<string, Tree> branches = 1;
}

message Holder {
  Ping ping = 1;
}
"#,
);

/// Encodes a field key followed by `value`, the encoded value of a field.
fn wire_field(number: u32, wire_type: WireType, value: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    encode_key(number, wire_type, &mut bytes);
    bytes.extend_from_slice(value);
    bytes
}

/// Encodes a varint field.
fn varint_field(number: u32, value: u64) -> Vec<u8> {
    let mut varint = Vec::new();
    encode_varint(value, &mut varint);
    wire_field(number, WireType::Varint, &varint)
}

/// Encodes a length-delimited field.
fn len_field(number: u32, payload: &[u8]) -> Vec<u8> {
    let mut value = Vec::new();
    encode_varint(payload.len() as u64, &mut value);
    value.extend_from_slice(payload);
    wire_field(number, WireType::LengthDelimited, &value)
}

/// Resolves imports among in-memory sources, by file name, then among the bundled well-known
/// types.
struct SourceResolver(Vec<(String, String)>);

impl FileResolver for SourceResolver {
    fn open_file(&self, name: &str) -> Result<File, protox::Error> {
        match self.0.iter().find(|(file, _)| file == name) {
            Some((file, source)) => File::from_source(file, source),
            None => Err(protox::Error::file_not_found(name)),
        }
    }
}

/// Compiles `sources`, pairs of a file name and its source, with source code info into one
/// descriptor pool.
fn compile(sources: &[(&str, &str)]) -> VortexResult<DescriptorPool> {
    let mut resolver = ChainFileResolver::new();
    resolver.add(SourceResolver(
        sources
            .iter()
            .map(|&(file, source)| (file.to_owned(), source.to_owned()))
            .collect(),
    ));
    resolver.add(GoogleFileResolver::new());
    let mut compiler = Compiler::with_file_resolver(resolver);
    compiler.include_imports(true).include_source_info(true);
    compiler
        .open_files(sources.iter().map(|(file, _)| file))
        .map_err(|err| vortex_err!("Invalid test schema: {err}"))?;
    Ok(compiler.descriptor_pool())
}

/// Returns the descriptor of the message `name` compiled from `sources`.
fn message(sources: &[(&str, &str)], name: &str) -> VortexResult<MessageDescriptor> {
    compile(sources)?
        .get_message_by_name(name)
        .ok_or_else(|| vortex_err!("Test schema does not define {name}"))
}

/// Builds a message of type `name`, from the pool of `descriptor`, with the given fields.
fn build(
    descriptor: &MessageDescriptor,
    name: &str,
    fields: impl IntoIterator<Item = (&'static str, Value)>,
) -> VortexResult<DynamicMessage> {
    let message_descriptor = descriptor
        .parent_pool()
        .get_message_by_name(name)
        .ok_or_else(|| vortex_err!("missing {name}"))?;
    let mut message = DynamicMessage::new(message_descriptor);
    for (field, value) in fields {
        message
            .try_set_field_by_name(field, value)
            .map_err(|err| vortex_err!("cannot set {field}: {err}"))?;
    }
    Ok(message)
}

/// Returns the protobuf dtype of the message `name` compiled from `sources`.
fn protobuf_dtype(
    sources: &[(&str, &str)],
    name: &str,
    nullability: Nullability,
) -> VortexResult<ExtDType<Protobuf>> {
    protobuf_dtype_with(sources, name, nullability, MappingOptions::default())
}

/// The mapping options that store unknown fields.
fn preserving() -> MappingOptions {
    MappingOptions {
        preserve_unknown_fields: true,
        ..MappingOptions::default()
    }
}

/// Returns the protobuf dtype of the message `name` compiled from `sources`, with the mapping
/// `options`.
fn protobuf_dtype_with(
    sources: &[(&str, &str)],
    name: &str,
    nullability: Nullability,
    options: MappingOptions,
) -> VortexResult<ExtDType<Protobuf>> {
    let metadata = ProtobufMetadata::try_new_with_options(&message(sources, name)?, options)?;
    ExtDType::try_new(metadata, nullability)
}

/// Decodes serialized messages, or nulls, into an array of `dtype`.
fn decode_rows(dtype: &ExtDType<Protobuf>, rows: &[Option<Vec<u8>>]) -> VortexResult<ArrayRef> {
    let mut decoder = ProtobufDecoder::new(dtype);
    for row in rows {
        match row {
            Some(message) => decoder.append_message(message)?,
            None => decoder.append_null()?,
        }
    }
    decoder.finish()
}

/// Encodes each message of a protobuf array, with `None` at null rows.
fn encode_rows(array: &ArrayRef) -> VortexResult<Vec<Option<Vec<u8>>>> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = encode_messages(array, &mut ctx)?;
    let validity = encoded.validity()?.execute_mask(encoded.len(), &mut ctx)?;
    Ok((0..encoded.len())
        .map(|row| {
            validity
                .value(row)
                .then(|| encoded.bytes_at(row).as_slice().to_vec())
        })
        .collect())
}

/// Lists the fields of a struct dtype, one per line.
fn describe_fields(dtype: &DType) -> String {
    let fields = dtype.as_struct_fields();
    fields
        .names()
        .iter()
        .zip(fields.fields())
        .map(|(name, dtype)| format!("{name}: {dtype}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Returns the protobuf dtype of `test.AllTypes`.
fn all_types_dtype(nullability: Nullability) -> VortexResult<ExtDType<Protobuf>> {
    protobuf_dtype(&[TEST_PROTO, UNRELATED_PROTO], "test.AllTypes", nullability)
}

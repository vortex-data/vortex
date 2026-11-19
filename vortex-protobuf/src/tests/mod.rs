// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared schemas and helpers for the crate's tests.

mod arrow;
mod codec;
mod editions;
mod file;
mod mapping;
mod metadata;

use std::sync::LazyLock;

use prost_reflect::DescriptorPool;
use prost_reflect::DynamicMessage;
use prost_reflect::MessageDescriptor;
use prost_reflect::Value;
use protox::Compiler;
use protox::file::ChainFileResolver;
use protox::file::File;
use protox::file::FileResolver;
use protox::file::GoogleFileResolver;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::Protobuf;
use crate::ProtobufMetadata;

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
    let metadata = ProtobufMetadata::try_new(&message(sources, name)?)?;
    ExtDType::try_new(metadata, nullability)
}

/// Returns the protobuf dtype of `test.AllTypes`.
fn all_types_dtype(nullability: Nullability) -> VortexResult<ExtDType<Protobuf>> {
    protobuf_dtype(&[TEST_PROTO, UNRELATED_PROTO], "test.AllTypes", nullability)
}

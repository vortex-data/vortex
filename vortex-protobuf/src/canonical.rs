// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The canonical descriptor set of a message schema, which the protobuf dtype stores in its
//! metadata.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use prost_reflect::FileDescriptor;
use prost_reflect::Kind;
use prost_reflect::MessageDescriptor;
use prost_types::DescriptorProto;
use prost_types::EnumDescriptorProto;
use prost_types::EnumOptions;
use prost_types::EnumValueDescriptorProto;
use prost_types::FieldDescriptorProto;
use prost_types::FieldOptions;
use prost_types::FileDescriptorProto;
use prost_types::FileDescriptorSet;
use prost_types::MessageOptions;
use prost_types::OneofDescriptorProto;

/// Returns the canonical descriptor set of messages of type `root`.
///
/// The set defines only what decoding and encoding the messages needs: the message types that are
/// reachable from `root` through fields, and the enum types of their fields. It holds the files
/// that define these types, each one after the files that it imports. Each file keeps its name,
/// package and syntax, and drops everything else:
///
/// - Other message and enum types, services and extensions. A message that is not reachable but
///   encloses a kept type keeps only its kept nested types.
/// - Imports of files that define none of the types that the file's kept fields use. The other
///   imports are ordered by name.
/// - Source code info, reserved names and ranges, extension ranges, JSON names and options,
///   except the options that change the wire format or the validity of a schema: `packed`,
///   `map_entry`, `message_set_wire_format` and `allow_alias`.
///
/// Message and enum types are ordered by name. Fields, oneofs and enum values keep their order.
pub(crate) fn canonical_descriptor_set(root: &MessageDescriptor) -> FileDescriptorSet {
    let reachable = Reachable::from_root(root);
    let mut file = Vec::with_capacity(reachable.files.len());
    let mut visited = BTreeSet::new();
    reachable.push_files(root.parent_file().name(), &mut visited, &mut file);
    FileDescriptorSet { file }
}

/// The types reachable from a root message, and the files that define them.
#[derive(Default)]
struct Reachable {
    /// The full names of the reachable message types.
    messages: BTreeSet<String>,
    /// The full names of the enum types of reachable fields.
    enums: BTreeSet<String>,
    /// The files that define reachable types, with the names of the other files whose types
    /// their reachable fields use.
    files: BTreeMap<String, (FileDescriptor, BTreeSet<String>)>,
}

impl Reachable {
    /// Finds the types reachable from `root`.
    ///
    /// Each message is visited once, so this terminates on recursive messages too.
    fn from_root(root: &MessageDescriptor) -> Self {
        let mut reachable = Self::default();
        let mut pending = vec![root.clone()];
        while let Some(message) = pending.pop() {
            if !reachable.messages.insert(message.full_name().to_owned()) {
                continue;
            }
            let file = message.parent_file();
            reachable.add_file(&file);
            for field in message.fields() {
                let used = match field.kind() {
                    Kind::Message(child) => {
                        let used = child.parent_file();
                        pending.push(child);
                        used
                    }
                    Kind::Enum(enum_type) => {
                        reachable.enums.insert(enum_type.full_name().to_owned());
                        enum_type.parent_file()
                    }
                    _ => continue,
                };
                reachable.add_file(&used);
                if used.name() != file.name()
                    && let Some((_, imports)) = reachable.files.get_mut(file.name())
                {
                    imports.insert(used.name().to_owned());
                }
            }
        }
        reachable
    }

    /// Records that `file` defines a reachable type.
    fn add_file(&mut self, file: &FileDescriptor) {
        self.files
            .entry(file.name().to_owned())
            .or_insert_with(|| (file.clone(), BTreeSet::new()));
    }

    /// Appends the canonical form of the file `name` to `files`, after its imports.
    fn push_files(
        &self,
        name: &str,
        visited: &mut BTreeSet<String>,
        files: &mut Vec<FileDescriptorProto>,
    ) {
        if !visited.insert(name.to_owned()) {
            return;
        }
        let Some((file, imports)) = self.files.get(name) else {
            return;
        };
        for import in imports {
            self.push_files(import, visited, files);
        }
        let proto = file.file_descriptor_proto();
        files.push(FileDescriptorProto {
            name: proto.name.clone(),
            package: proto.package.clone(),
            dependency: imports.iter().cloned().collect(),
            message_type: self.messages(&proto.message_type, proto.package()),
            enum_type: self.enums(&proto.enum_type, proto.package()),
            syntax: proto.syntax.clone(),
            ..Default::default()
        });
    }

    /// Returns the canonical forms of the kept `messages` declared in the scope `scope`.
    fn messages(&self, messages: &[DescriptorProto], scope: &str) -> Vec<DescriptorProto> {
        let mut kept = messages
            .iter()
            .filter_map(|message| self.message(message, scope))
            .collect::<Vec<_>>();
        kept.sort_by(|a, b| a.name().cmp(b.name()));
        kept
    }

    /// Returns the canonical form of `message`, declared in the scope `scope`, or `None` if
    /// neither it nor any type nested in it is kept.
    fn message(&self, message: &DescriptorProto, scope: &str) -> Option<DescriptorProto> {
        let full_name = full_name(scope, message.name());
        let nested_type = self.messages(&message.nested_type, &full_name);
        let enum_type = self.enums(&message.enum_type, &full_name);
        if !self.messages.contains(&full_name) {
            return (!nested_type.is_empty() || !enum_type.is_empty()).then(|| DescriptorProto {
                name: message.name.clone(),
                nested_type,
                enum_type,
                ..Default::default()
            });
        }
        let options = message.options.as_ref().map(|options| MessageOptions {
            message_set_wire_format: options.message_set_wire_format.filter(|&set| set),
            map_entry: options.map_entry.filter(|&entry| entry),
            ..Default::default()
        });
        Some(DescriptorProto {
            name: message.name.clone(),
            field: message.field.iter().map(canonical_field).collect(),
            nested_type,
            enum_type,
            oneof_decl: message
                .oneof_decl
                .iter()
                .map(|oneof| OneofDescriptorProto {
                    name: oneof.name.clone(),
                    options: None,
                })
                .collect(),
            options: options.filter(|options| *options != MessageOptions::default()),
            ..Default::default()
        })
    }

    /// Returns the canonical forms of the kept `enums` declared in the scope `scope`.
    fn enums(&self, enums: &[EnumDescriptorProto], scope: &str) -> Vec<EnumDescriptorProto> {
        let mut kept = enums
            .iter()
            .filter(|enum_type| self.enums.contains(&full_name(scope, enum_type.name())))
            .map(canonical_enum)
            .collect::<Vec<_>>();
        kept.sort_by(|a, b| a.name().cmp(b.name()));
        kept
    }
}

/// Returns the canonical form of a field of a kept message.
fn canonical_field(field: &FieldDescriptorProto) -> FieldDescriptorProto {
    FieldDescriptorProto {
        json_name: None,
        options: field
            .options
            .as_ref()
            .and_then(|options| options.packed)
            .map(|packed| FieldOptions {
                packed: Some(packed),
                ..Default::default()
            }),
        ..field.clone()
    }
}

/// Returns the canonical form of a kept enum.
fn canonical_enum(enum_type: &EnumDescriptorProto) -> EnumDescriptorProto {
    EnumDescriptorProto {
        name: enum_type.name.clone(),
        value: enum_type
            .value
            .iter()
            .map(|value| EnumValueDescriptorProto {
                name: value.name.clone(),
                number: value.number,
                options: None,
            })
            .collect(),
        options: enum_type
            .options
            .as_ref()
            .and_then(|options| options.allow_alias)
            .filter(|&allow_alias| allow_alias)
            .map(|allow_alias| EnumOptions {
                allow_alias: Some(allow_alias),
                ..Default::default()
            }),
        ..Default::default()
    }
}

/// Returns the full name of the type `name` declared in the scope `scope`, a package or message.
fn full_name(scope: &str, name: &str) -> String {
    if scope.is_empty() {
        name.to_owned()
    } else {
        format!("{scope}.{name}")
    }
}

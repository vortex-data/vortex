// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Metadata of the protobuf extension dtype: the message schema and the storage mapping version.

use std::collections::BTreeSet;
use std::fmt;
use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use prost::Message;
use prost_reflect::DescriptorPool;
use prost_reflect::Kind;
use prost_reflect::MessageDescriptor;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::canonical::canonical_descriptor_set;

/// The storage mapping version that this crate writes.
pub const MAPPING_VERSION: u32 = 1;

/// The schema of a protobuf extension dtype.
///
/// The metadata names a root message and holds a canonical, serialized
/// [`FileDescriptorSet`](prost_types::FileDescriptorSet) that defines it. The descriptor set
/// defines only the message types that are reachable from the root message through fields, and
/// the enum types of their fields. It drops the other types of their files, services, extensions,
/// unused imports, source code info and the options that do not change the wire format, such as
/// `java_package` or `deprecated`. So the same message schema always gives the same bytes,
/// independent of how the caller's descriptor set was produced or ordered and of what else its
/// files declare.
///
/// Equality and hashing use only the serialized parts: the mapping version, the root message name
/// and the descriptor set bytes. The parsed [`MessageDescriptor`] is kept for fast access.
#[derive(Clone)]
pub struct ProtobufMetadata {
    /// The version of the mapping from the message schema to the Vortex storage dtype.
    mapping_version: u32,
    /// The root message, resolved in a pool built from `descriptor_set`.
    descriptor: MessageDescriptor,
    /// The serialized descriptor set that defines the root message.
    descriptor_set: Arc<[u8]>,
}

/// The serialized form of [`ProtobufMetadata`].
#[derive(Clone, PartialEq, Message)]
pub(crate) struct SerializedMetadata {
    /// See [`ProtobufMetadata::mapping_version`].
    #[prost(uint32, tag = "1")]
    pub(crate) mapping_version: u32,
    /// See [`ProtobufMetadata::message_name`].
    #[prost(string, tag = "2")]
    pub(crate) message_name: String,
    /// See [`ProtobufMetadata::descriptor_set`].
    #[prost(bytes = "vec", tag = "3")]
    pub(crate) descriptor_set: Vec<u8>,
}

impl ProtobufMetadata {
    /// Creates metadata for messages of type `descriptor` with the current [`MAPPING_VERSION`].
    ///
    /// # Errors
    ///
    /// Returns an error if the canonical descriptor set does not define the message, which can
    /// only happen if the descriptor's pool is inconsistent.
    pub fn try_new(descriptor: &MessageDescriptor) -> VortexResult<Self> {
        let descriptor_set = canonical_descriptor_set(descriptor).encode_to_vec();
        Self::from_parts(
            MAPPING_VERSION,
            descriptor.full_name(),
            descriptor_set.into(),
        )
    }

    /// Resolves the root message `message_name` in the serialized `descriptor_set`.
    fn from_parts(
        mapping_version: u32,
        message_name: &str,
        descriptor_set: Arc<[u8]>,
    ) -> VortexResult<Self> {
        let pool = DescriptorPool::decode(descriptor_set.as_ref())
            .map_err(|err| vortex_err!("Invalid protobuf descriptor set: {err}"))?;
        let descriptor = pool.get_message_by_name(message_name).ok_or_else(|| {
            vortex_err!("Protobuf descriptor set does not define the root message {message_name:?}")
        })?;
        Ok(Self {
            mapping_version,
            descriptor,
            descriptor_set,
        })
    }

    /// Decodes metadata written by [`Self::serialize`].
    ///
    /// The descriptor set is kept as stored, so the metadata serializes back to the same bytes.
    pub(crate) fn deserialize(bytes: &[u8]) -> VortexResult<Self> {
        let serialized = SerializedMetadata::decode(bytes).map_err(|err| {
            VortexError::from(err).with_context("Invalid protobuf extension metadata")
        })?;
        Self::from_parts(
            serialized.mapping_version,
            &serialized.message_name,
            serialized.descriptor_set.into(),
        )
    }

    /// Encodes the metadata for storage.
    pub(crate) fn serialize(&self) -> Vec<u8> {
        SerializedMetadata {
            mapping_version: self.mapping_version,
            message_name: self.message_name().to_owned(),
            descriptor_set: self.descriptor_set.to_vec(),
        }
        .encode_to_vec()
    }

    /// Checks that this crate supports the mapping version and that the message is not recursive.
    pub(crate) fn validate(&self) -> VortexResult<()> {
        if self.mapping_version != MAPPING_VERSION {
            vortex_bail!(
                "Unsupported protobuf storage mapping version {}, expected {MAPPING_VERSION}",
                self.mapping_version
            );
        }
        ensure_not_recursive(&self.descriptor, &mut Vec::new(), &mut BTreeSet::new())
    }

    /// The version of the mapping from the message schema to the Vortex storage dtype.
    pub fn mapping_version(&self) -> u32 {
        self.mapping_version
    }

    /// The fully qualified name of the root message, such as `my.package.Event`.
    pub fn message_name(&self) -> &str {
        self.descriptor.full_name()
    }

    /// The descriptor of the root message.
    ///
    /// The descriptor belongs to a pool built from [`descriptor_set`](Self::descriptor_set), not
    /// to the pool of the descriptor that created the metadata. Descriptors of different pools are
    /// never equal, and neither are messages of their types, even when the schemas match. Compare
    /// such messages by their encoded bytes, or decode them with this descriptor.
    pub fn message_descriptor(&self) -> &MessageDescriptor {
        &self.descriptor
    }

    /// The serialized [`FileDescriptorSet`](prost_types::FileDescriptorSet) that defines the root
    /// message.
    pub fn descriptor_set(&self) -> &[u8] {
        &self.descriptor_set
    }
}

impl PartialEq for ProtobufMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.mapping_version == other.mapping_version
            && self.message_name() == other.message_name()
            && self.descriptor_set == other.descriptor_set
    }
}

impl Eq for ProtobufMetadata {}

impl Hash for ProtobufMetadata {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.mapping_version.hash(state);
        self.message_name().hash(state);
        self.descriptor_set.hash(state);
    }
}

impl fmt::Debug for ProtobufMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProtobufMetadata")
            .field("mapping_version", &self.mapping_version)
            .field("message_name", &self.message_name())
            .field(
                "descriptor_set",
                &DescriptorSetSummary(&self.descriptor_set),
            )
            .finish()
    }
}

/// Formats a serialized descriptor set by its length and a hash of its bytes, which tells two
/// descriptor sets apart without printing them.
struct DescriptorSetSummary<'a>(&'a [u8]);

impl fmt::Debug for DescriptorSetSummary<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut hasher = DefaultHasher::new();
        self.0.hash(&mut hasher);
        write!(f, "<{} bytes, hash {:016x}>", self.0.len(), hasher.finish())
    }
}

impl fmt::Display for ProtobufMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message_name())
    }
}

/// Fails if `message` contains itself, directly or through other messages.
///
/// `path` holds the messages from the root down to `message`, and `checked` holds messages whose
/// fields are already known to be free of cycles.
fn ensure_not_recursive(
    message: &MessageDescriptor,
    path: &mut Vec<MessageDescriptor>,
    checked: &mut BTreeSet<String>,
) -> VortexResult<()> {
    if checked.contains(message.full_name()) {
        return Ok(());
    }
    if let Some(start) = path.iter().position(|ancestor| ancestor == message) {
        let cycle = path[start..]
            .iter()
            .chain([message])
            .map(MessageDescriptor::full_name)
            .collect::<Vec<_>>()
            .join(" -> ");
        vortex_bail!("Recursive protobuf messages are not supported: {cycle}");
    }

    path.push(message.clone());
    for field in message.fields() {
        if let Kind::Message(child) = field.kind() {
            ensure_not_recursive(&child, path, checked)?;
        }
    }
    path.pop();
    checked.insert(message.full_name().to_owned());
    Ok(())
}

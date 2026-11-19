// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Metadata of the protobuf extension dtype: the message schema, the storage mapping version and
//! the mapping options.

use std::fmt;
use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use prost::Message;
use prost_reflect::MessageDescriptor;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::MappingOptions;
use crate::canonical::canonical_descriptor_set;
use crate::mapping::MessagePlan;
use crate::options::SerializedOptions;
use crate::pool::descriptor_pool;

/// The storage mapping version that this crate writes.
///
/// Version 1 maps every schema as version 2 does with the default [`MappingOptions`], and has no
/// options. This crate reads both versions, and in both rejects a schema whose storage would have
/// more than 100,000 fields.
pub const MAPPING_VERSION: u32 = 2;

/// The first mapping version that has options.
const OPTIONS_VERSION: u32 = 2;

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
/// The [`MappingOptions`] record the choices of the storage mapping, so the storage dtype is a
/// function of the metadata.
///
/// Equality and hashing use only the serialized parts: the mapping version, the options, the root
/// message name and the descriptor set bytes. The parsed [`MessageDescriptor`] and the storage
/// plan are kept for fast access.
#[derive(Clone)]
pub struct ProtobufMetadata {
    /// The version of the mapping from the message schema to the Vortex storage dtype.
    mapping_version: u32,
    /// The choices of the storage mapping.
    options: MappingOptions,
    /// The root message, resolved in a pool built from `descriptor_set`.
    descriptor: MessageDescriptor,
    /// The serialized descriptor set that defines the root message.
    descriptor_set: Arc<[u8]>,
    /// How the root message is stored.
    plan: Arc<MessagePlan>,
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
    /// See [`ProtobufMetadata::options`], absent for the default options and in version 1.
    #[prost(message, optional, tag = "4")]
    pub(crate) options: Option<SerializedOptions>,
}

impl ProtobufMetadata {
    /// Creates metadata for messages of type `descriptor` with the current [`MAPPING_VERSION`]
    /// and the default [`MappingOptions`].
    ///
    /// # Errors
    ///
    /// Returns an error if the message is recursive, which needs the
    /// [`max_recursion_depth`](MappingOptions::max_recursion_depth) option, if its storage would
    /// have more than 100,000 fields, or if the canonical descriptor set does not define the
    /// message, which can only happen if the descriptor's pool is inconsistent.
    pub fn try_new(descriptor: &MessageDescriptor) -> VortexResult<Self> {
        Self::try_new_with_options(descriptor, MappingOptions::default())
    }

    /// Creates metadata for messages of type `descriptor` with the current [`MAPPING_VERSION`]
    /// and the given options.
    ///
    /// # Errors
    ///
    /// Returns an error if the options are out of range, if the message is recursive and the
    /// options set no [`max_recursion_depth`](MappingOptions::max_recursion_depth), if its storage
    /// would have more than 100,000 fields, or if the canonical descriptor set does not define the
    /// message, which can only happen if the descriptor's pool is inconsistent.
    pub fn try_new_with_options(
        descriptor: &MessageDescriptor,
        options: MappingOptions,
    ) -> VortexResult<Self> {
        let descriptor_set = canonical_descriptor_set(descriptor).encode_to_vec();
        Self::from_parts(
            MAPPING_VERSION,
            options,
            descriptor.full_name(),
            &descriptor_set,
        )
    }

    /// Resolves the root message `message_name` in the serialized `descriptor_set` and plans its
    /// storage.
    ///
    /// The pool comes from the process-wide cache, so the same descriptor set is parsed once.
    fn from_parts(
        mapping_version: u32,
        options: MappingOptions,
        message_name: &str,
        descriptor_set: &[u8],
    ) -> VortexResult<Self> {
        if !(1..=MAPPING_VERSION).contains(&mapping_version) {
            vortex_bail!(
                "Unsupported protobuf storage mapping version {mapping_version}, expected 1 to \
                 {MAPPING_VERSION}"
            );
        }
        options.validate()?;
        let (descriptor_set, pool) = descriptor_pool(descriptor_set)?;
        let descriptor = pool.get_message_by_name(message_name).ok_or_else(|| {
            vortex_err!("Protobuf descriptor set does not define the root message {message_name:?}")
        })?;
        let plan = MessagePlan::try_new(&descriptor, &options)?;
        Ok(Self {
            mapping_version,
            options,
            descriptor,
            descriptor_set,
            plan: Arc::new(plan),
        })
    }

    /// Decodes metadata written by [`Self::serialize`].
    ///
    /// The descriptor set is kept as stored, so the metadata serializes back to the same bytes.
    pub(crate) fn deserialize(bytes: &[u8]) -> VortexResult<Self> {
        let serialized = SerializedMetadata::decode(bytes).map_err(|err| {
            VortexError::from(err).with_context("Invalid protobuf extension metadata")
        })?;
        let options = match &serialized.options {
            Some(_) if serialized.mapping_version < OPTIONS_VERSION => vortex_bail!(
                "Protobuf storage mapping version {} has no options",
                serialized.mapping_version
            ),
            Some(options) => MappingOptions::deserialize(options)?,
            None => MappingOptions::default(),
        };
        Self::from_parts(
            serialized.mapping_version,
            options,
            &serialized.message_name,
            &serialized.descriptor_set,
        )
    }

    /// Encodes the metadata for storage.
    pub(crate) fn serialize(&self) -> Vec<u8> {
        SerializedMetadata {
            mapping_version: self.mapping_version,
            message_name: self.message_name().to_owned(),
            descriptor_set: self.descriptor_set.to_vec(),
            options: (self.options != MappingOptions::default()).then(|| self.options.serialize()),
        }
        .encode_to_vec()
    }

    /// The version of the mapping from the message schema to the Vortex storage dtype.
    pub fn mapping_version(&self) -> u32 {
        self.mapping_version
    }

    /// The choices of the storage mapping.
    pub fn options(&self) -> &MappingOptions {
        &self.options
    }

    /// How the root message is stored.
    pub(crate) fn plan(&self) -> &MessagePlan {
        &self.plan
    }

    /// The fully qualified name of the root message, such as `my.package.Event`.
    pub fn message_name(&self) -> &str {
        self.descriptor.full_name()
    }

    /// The descriptor of the root message.
    ///
    /// The descriptor belongs to a pool built from [`descriptor_set`](Self::descriptor_set), not
    /// to the pool of the descriptor that created the metadata. Metadata of the same schema share
    /// that pool while a process-wide cache holds it. Descriptors of different pools are never
    /// equal, and neither are messages of their types, even when the schemas match. Compare such
    /// messages by their encoded bytes, or decode them with this descriptor.
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
            && self.options == other.options
            && self.message_name() == other.message_name()
            && self.descriptor_set == other.descriptor_set
    }
}

impl Eq for ProtobufMetadata {}

impl Hash for ProtobufMetadata {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.mapping_version.hash(state);
        self.options.hash(state);
        self.message_name().hash(state);
        self.descriptor_set.hash(state);
    }
}

impl fmt::Debug for ProtobufMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProtobufMetadata")
            .field("mapping_version", &self.mapping_version)
            .field("options", &self.options)
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

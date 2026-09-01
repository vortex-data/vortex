// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::VTable;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;

/// Reference-counted array plugin.
pub type ArrayPluginRef = Arc<dyn ArrayPlugin>;

/// The wire representation produced by an in-memory array's serializer.
///
/// A serializer may reuse the in-memory array's buffers and children with [`Self::from_array`],
/// or return different parts when an older wire representation requires a lossless structural
/// downgrade.
#[derive(Clone, Debug)]
pub struct ArraySerialization {
    /// The concrete array ID to write on the wire.
    pub serialized_id: ArrayId,
    /// Encoding-specific metadata written into the array node.
    pub metadata: Vec<u8>,
    /// Top-level buffers written for this array node.
    pub buffers: Vec<ByteBuffer>,
    /// Child arrays to serialize recursively.
    pub children: Vec<ArrayRef>,
}

impl ArraySerialization {
    /// Create a wire representation from an ID, metadata, buffers, and children.
    pub fn new(
        serialized_id: ArrayId,
        metadata: Vec<u8>,
        buffers: Vec<ByteBuffer>,
        children: Vec<ArrayRef>,
    ) -> Self {
        Self {
            serialized_id,
            metadata,
            buffers,
            children,
        }
    }

    /// Reuse an in-memory array's buffers and children with the supplied serialized metadata.
    pub fn from_array(serialized_id: ArrayId, array: &ArrayRef, metadata: Vec<u8>) -> Self {
        Self::new(serialized_id, metadata, array.buffers(), array.children())
    }
}

/// The borrowed wire components passed to an array deserializer.
pub struct ArrayDeserialization<'a> {
    /// The exact array ID found on the wire.
    pub serialized_id: ArrayId,
    /// The logical dtype supplied by the containing format.
    pub dtype: &'a DType,
    /// The logical array length supplied by the containing format.
    pub len: usize,
    /// Encoding-specific metadata from the array node.
    pub metadata: &'a [u8],
    /// Top-level buffers referenced by the array node.
    pub buffers: &'a [BufferHandle],
    /// Lazily decoded child arrays referenced by the array node.
    pub children: &'a dyn ArrayChildren,
}

impl<'a> ArrayDeserialization<'a> {
    /// Create borrowed deserialization input from a wire ID and its serialized components.
    pub fn new(
        serialized_id: ArrayId,
        dtype: &'a DType,
        len: usize,
        metadata: &'a [u8],
        buffers: &'a [BufferHandle],
        children: &'a dyn ArrayChildren,
    ) -> Self {
        Self {
            serialized_id,
            dtype,
            len,
            metadata,
            buffers,
            children,
        }
    }
}

/// Registry trait for serializing and deserializing an in-memory array representation.
///
/// A plugin has one [`id`](Self::id) for the in-memory representation and one or more
/// [`serialized_ids`](Self::serialized_ids) for wire representations. Its serializer chooses the
/// wire representation, and the serialization context validates that the chosen ID is permitted
/// before it is written.
///
/// Every serialized ID is also registered for deserialization. A current plugin may therefore
/// deserialize several historical IDs into the same in-memory representation. A reader that
/// predates a newer ID has no registration for it and reports it as unknown instead of silently
/// interpreting an unsupported representation.
pub trait ArrayPlugin: 'static + Send + Sync {
    /// Returns the ID of the in-memory array representation handled by this plugin.
    fn id(&self) -> ArrayId;

    /// Returns the serialized array IDs understood by this plugin, ordered oldest to newest.
    ///
    /// The default uses the in-memory ID as the sole wire ID. Override this for an in-memory array
    /// that has multiple serialized variants. IDs retained only for reading may also be included;
    /// the single serializer need not select them.
    fn serialized_ids(&self) -> Vec<ArrayId> {
        vec![self.id()]
    }

    /// Serialize `array` to its wire representation.
    ///
    /// This function is called only for arrays whose in-memory encoding matches [`id`](Self::id).
    /// The returned ID must be declared by [`serialized_ids`](Self::serialized_ids). Return
    /// `Ok(None)` when the array cannot be serialized.
    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>>;

    /// Deserialize one recognized wire representation into the current in-memory array.
    ///
    /// `serialized_id` identifies the exact representation encountered on disk. The returned
    /// array does not necessarily have to use this plugin's in-memory ID; this supports legacy
    /// representations that are normalized into another current in-memory array. Implementations
    /// must validate the contract of that exact ID rather than accepting every form understood by
    /// the current in-memory representation under an older ID.
    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef>;

    /// Can this plugin emit an array with the given encoding.
    ///
    /// By default, this is just the [ID][Self::id] of the plugin, but
    /// can be overridden if this plugin instance supports reading/writing multiple arrays.
    fn is_supported_encoding(&self, id: &ArrayId) -> bool {
        self.id() == *id
    }
}

impl Debug for dyn ArrayPlugin {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ArrayPlugin").field(&self.id()).finish()
    }
}

impl<V: VTable> ArrayPlugin for V {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure!(
            self.id() == array.encoding_id(),
            "array plugin {} cannot serialize in-memory array {}",
            self.id(),
            array.encoding_id(),
        );
        Ok(V::serialize(array.as_::<V>(), session)?
            .map(|metadata| ArraySerialization::from_array(self.id(), array, metadata)))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            self.id() == parts.serialized_id,
            "array plugin {} does not recognize serialized ID {}",
            self.id(),
            parts.serialized_id,
        );
        Ok(Array::<V>::try_from_parts(V::deserialize(
            self,
            parts.dtype,
            parts.len,
            parts.metadata,
            parts.buffers,
            parts.children,
            session,
        )?)?
        .into_array())
    }
}

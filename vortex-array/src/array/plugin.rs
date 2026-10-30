// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
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
/// or return the wire representation of another encoding. The node must hold the logical values
/// of the array under the dtype supplied by the containing format.
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
/// A plugin writes one in-memory encoding, identified by [`id`](Self::id). It declares the wire
/// IDs that it reads and writes separately, so it can write another encoding's wire form without
/// replacing that encoding's reader. The serialization context validates that the chosen wire ID
/// is permitted before it is written.
///
/// A plugin can retain historical read IDs after it stops writing them. A reader that predates a
/// newer ID has no registration for it and reports it as unknown.
pub trait ArrayPlugin: 'static + Send + Sync {
    /// Returns the ID of the in-memory encoding that this plugin writes.
    fn id(&self) -> ArrayId;

    /// Returns the wire IDs that [`deserialize`](Self::deserialize) accepts, oldest to newest.
    ///
    /// [`ArraySession::register`](crate::session::ArraySession::register) makes this plugin the
    /// reader of each ID. An ID can stay in this list after the plugin stops writing it, so that
    /// old files stay readable.
    fn read_ids(&self) -> Vec<ArrayId>;

    /// Returns the wire IDs that [`serialize`](Self::serialize) may return.
    ///
    /// An ID in this list does not have to be in [`read_ids`](Self::read_ids). A plugin can write
    /// its arrays in the wire form of another encoding. The plugin that owns that form reads it.
    /// The node must hold the logical values of the array under the dtype of the array.
    fn write_ids(&self) -> Vec<ArrayId>;

    /// Serialize `array` to its wire representation.
    ///
    /// This function is called only for arrays whose in-memory encoding matches [`id`](Self::id).
    /// The returned ID must be declared by [`write_ids`](Self::write_ids). Return
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

    fn read_ids(&self) -> Vec<ArrayId> {
        vec![self.id()]
    }

    fn write_ids(&self) -> Vec<ArrayId> {
        vec![self.id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure_eq!(
            self.id(),
            array.encoding_id(),
            "array plugin cannot serialize an in-memory array of another encoding",
        );
        Ok(V::serialize(array.as_::<V>(), session)?
            .map(|metadata| ArraySerialization::from_array(self.id(), array, metadata)))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            self.id(),
            parts.serialized_id,
            "array plugin does not recognize the serialized ID",
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

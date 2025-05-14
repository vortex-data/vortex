use std::fmt::Debug;

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::serde::ArrayParts;
use crate::vtable::{NotSupported, VTable};
use crate::{ArrayContext, DeserializeMetadata, EmptyMetadata, SerializeMetadata};

/// VTable for assisting with the serialization and deserialiation of arrays.
///
/// It is required to implement this vtable in order to support:
///  * Serialization to disk or over IPC.
///  * Import/export over FFI.
pub trait SerdeVTable<V: VTable> {
    type Metadata: Debug + SerializeMetadata + DeserializeMetadata;

    /// Exports the metadata for the array.
    ///
    /// All other parts of the array are exported using the [`crate::vtable::VisitorVTable`].
    ///
    /// * If the array does not require serialized metadata, it should return
    ///   [`crate::metadata::EmptyMetadata`].
    /// * If the array does not support serialization, it should return `None`.
    fn metadata(array: &V::Array) -> VortexResult<Option<Self::Metadata>>;

    /// Build an array from its given parts.
    fn build(
        encoding: &V::Encoding,
        dtype: DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<V::Array>;
}

impl<V: VTable> SerdeVTable<V> for NotSupported {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &V::Array) -> VortexResult<Option<Self::Metadata>> {
        Ok(None)
    }

    fn build(
        encoding: &V::Encoding,
        _dtype: DType,
        _len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
        _children: &[ArrayParts],
        _ctx: &ArrayContext,
    ) -> VortexResult<V::Array> {
        vortex_bail!("Serde not supported by {} encoding", V::id(encoding));
    }
}

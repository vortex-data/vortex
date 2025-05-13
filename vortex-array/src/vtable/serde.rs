use std::fmt::Debug;

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::serde::ArrayParts;
use crate::vtable::{NotSupported, VTable};
use crate::{ArrayContext, DeserializeMetadata, EmptyMetadata, SerializeMetadata};

/// VTable for implementing marshalling of arrays.
///
/// This is required to be implemented in order to support:
///  * Serialization to disk or over IPC.
///  * Import/export over FFI.
pub trait SerdeVTable<V: VTable> {
    type Metadata: Debug + SerializeMetadata + DeserializeMetadata;

    /// Returns the metadata for the given array.
    ///
    /// If `None` is returned, the array does not support serialization.
    fn metadata(array: &V::Array) -> Option<Self::Metadata>;

    /// Unmarshall an array from the given [`ArrayParts`] and [`ArrayContext`].
    /// The array parts must be valid for the given encoding.
    // TODO(ngates): rename to `unmarshal`
    fn decode(
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

    fn metadata(_array: &V::Array) -> Option<Self::Metadata> {
        None
    }

    fn decode(
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

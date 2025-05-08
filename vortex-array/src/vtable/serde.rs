use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::serde::ArrayParts;
use crate::vtable::VTable;
use crate::{Array, ArrayContext, Canonical};

/// VTable for implementing serialization and deserialization of arrays.
pub trait SerdeVTable<V: VTable> {
    type Metadata;

    /// Encodes a canonical array using this encoding.
    fn encode(
        encoding: &V::Encoding,
        canonical: Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<V::Array>;

    /// Decode an array from the given [`ArrayParts`] and [`ArrayContext`].
    /// The array parts must be valid for the given encoding.
    fn decode(
        encoding: &V::Encoding,
        dtype: DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<V::Array>;
}

impl<V: VTable> SerdeVTable<V> for () {
    type Metadata = ();

    fn encode(
        encoding: &V::Encoding,
        _canonical: Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<V::Array> {
        vortex_bail!("Serde not supported by {} encoding", V::id(encoding));
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

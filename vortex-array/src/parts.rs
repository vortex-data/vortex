use std::fmt::{Debug, Formatter};

use flatbuffers::Follow;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexResult};
use vortex_flatbuffers::{array as fba, FlatBuffer};

use crate::{Array, ContextRef};

/// [`ArrayParts`] represents the information from an [`Array`] that makes up the serialized
/// form. For example, it uses stores integer encoding IDs rather than a reference to an encoding
/// vtable, and it doesn't store any [`DType`] information.
///
/// An [`ArrayParts`] can be fully decoded into an [`Array`] using the `decode` function.
pub struct ArrayParts {
    // Typed as fb::Array
    flatbuffer: FlatBuffer,
    flatbuffer_loc: usize,
    buffers: Vec<ByteBuffer>,
}

impl Debug for ArrayParts {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayParts")
            .field("flatbuffer", &self.flatbuffer.len())
            .field("flatbuffer_loc", &self.flatbuffer_loc)
            .field("buffers", &self.buffers.len())
            .finish()
    }
}

impl ArrayParts {
    /// Creates a new [`ArrayParts`] from a flatbuffer view.
    ///
    /// ## Panics
    ///
    /// This function will panic if the flatbuffer is not contained within the given [`FlatBuffer`].
    pub fn new(array: fba::Array, flatbuffer: FlatBuffer, buffers: Vec<ByteBuffer>) -> Self {
        // We ensure that the flatbuffer given to us does indeed match that of the ByteBuffer
        if flatbuffer
            .as_ref()
            .as_slice()
            .subslice_range(array._tab.buf())
            != Some(0..flatbuffer.len())
        {
            vortex_panic!("Array flatbuffer is not contained within the buffer");
        }
        Self {
            flatbuffer,
            flatbuffer_loc: array._tab.loc(),
            buffers,
        }
    }

    /// Decode an [`ArrayParts`] into an [`Array`].
    pub fn decode(self, ctx: ContextRef, dtype: DType, len: usize) -> VortexResult<Array> {
        Array::try_new_viewed(
            ctx,
            dtype,
            len,
            self.flatbuffer,
            // SAFETY: ArrayComponents guarantees the buffers are valid.
            |buf| unsafe { Ok(fba::ArrayNode::follow(buf, self.flatbuffer_loc)) },
            self.buffers,
        )
    }
}

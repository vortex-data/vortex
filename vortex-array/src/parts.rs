use std::fmt::{Debug, Formatter};

use flatbuffers::{FlatBufferBuilder, Follow, WIPOffset};
use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexExpect, VortexResult};
use vortex_flatbuffers::{
    array as fba, FlatBuffer, FlatBufferRoot, WriteFlatBuffer, WriteFlatBufferExt,
};

use crate::stats::ArrayStatistics;
use crate::{ArrayData, ContextRef};

/// [`ArrayParts`] represents the information from an [`ArrayData`] that makes up the serialized
/// form. For example, it uses stores integer encoding IDs rather than a reference to an encoding
/// vtable, and it doesn't store any [`DType`] information.
///
/// An [`ArrayParts`] can be fully decoded into an [`ArrayData`] using the `decode` function.
pub struct ArrayParts {
    // TODO(ngates): I think we should remove this. It's not required in the serialized form.
    row_count: usize,
    // Typed as fb::Array
    flatbuffer: FlatBuffer,
    flatbuffer_loc: usize,
    buffers: Vec<ByteBuffer>,
}

impl Debug for ArrayParts {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayParts")
            .field("row_count", &self.row_count)
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
    pub fn new(
        row_count: usize,
        array: fba::Array,
        flatbuffer: FlatBuffer,
        buffers: Vec<ByteBuffer>,
    ) -> Self {
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
            row_count,
            flatbuffer,
            flatbuffer_loc: array._tab.loc(),
            buffers,
        }
    }

    /// Decode an [`ArrayParts`] into an [`ArrayData`].
    pub fn decode(self, ctx: ContextRef, dtype: DType) -> VortexResult<ArrayData> {
        ArrayData::try_new_viewed(
            ctx,
            dtype,
            self.row_count,
            self.flatbuffer,
            // SAFETY: ArrayComponents guarantees the buffers are valid.
            |buf| unsafe { Ok(fba::Array::follow(buf, self.flatbuffer_loc)) },
            self.buffers,
        )
    }
}

/// Convert an [`ArrayData`] into [`ArrayParts`].
impl From<ArrayData> for ArrayParts {
    fn from(array: ArrayData) -> Self {
        let flatbuffer = ArrayPartsFlatBuffer {
            array: &array,
            buffer_idx: 0,
        }
        .write_flatbuffer_bytes();
        let mut buffers: Vec<ByteBuffer> = vec![];
        for child in array.depth_first_traversal() {
            for buffer in child.byte_buffers() {
                buffers.push(buffer);
            }
        }
        Self {
            row_count: array.len(),
            flatbuffer,
            flatbuffer_loc: 0,
            buffers,
        }
    }
}

/// A utility struct for creating an [`fba::Array`] flatbuffer.
pub struct ArrayPartsFlatBuffer<'a> {
    array: &'a ArrayData,
    buffer_idx: u16,
}

impl<'a> ArrayPartsFlatBuffer<'a> {
    pub fn new(array: &'a ArrayData) -> Self {
        Self {
            array,
            buffer_idx: 0,
        }
    }
}

impl FlatBufferRoot for ArrayPartsFlatBuffer<'_> {}

impl WriteFlatBuffer for ArrayPartsFlatBuffer<'_> {
    type Target<'t> = fba::Array<'t>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let encoding = self.array.encoding().id().code();
        let metadata = self
            .array
            .metadata_bytes()
            .vortex_expect("IPCArray is missing metadata during serialization");
        let metadata = Some(fbb.create_vector(metadata.as_ref()));

        // Assign buffer indices for all child arrays.
        let nbuffers = u16::try_from(self.array.nbuffers())
            .vortex_expect("Array can have at most u16::MAX buffers");
        let child_buffer_idx = self.buffer_idx + nbuffers;

        let children = self
            .array
            .children()
            .iter()
            .scan(child_buffer_idx, |buffer_idx, child| {
                // Update the number of buffers required.
                let msg = ArrayPartsFlatBuffer {
                    array: child,
                    buffer_idx: *buffer_idx,
                }
                .write_flatbuffer(fbb);
                *buffer_idx = u16::try_from(child.cumulative_nbuffers())
                    .ok()
                    .and_then(|nbuffers| nbuffers.checked_add(*buffer_idx))
                    .vortex_expect("Too many buffers (u16) for ArrayData");
                Some(msg)
            })
            .collect_vec();
        let children = Some(fbb.create_vector(&children));

        let buffers = Some(fbb.create_vector_from_iter((0..nbuffers).map(|i| i + self.buffer_idx)));

        let stats = Some(self.array.statistics().write_flatbuffer(fbb));

        fba::Array::create(
            fbb,
            &fba::ArrayArgs {
                encoding,
                metadata,
                children,
                buffers,
                stats,
            },
        )
    }
}

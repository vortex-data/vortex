use std::fmt::{Debug, Formatter};

use flatbuffers::{FlatBufferBuilder, WIPOffset};
use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_flatbuffers::owned::array::OwnedArray;
use vortex_flatbuffers::{array as fba, FlatBufferRoot, WriteFlatBuffer, WriteFlatBufferExt};

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
    array: OwnedArray,
    buffers: Vec<ByteBuffer>,
}

impl Debug for ArrayParts {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayParts")
            .field("row_count", &self.row_count)
            .field("buffers", &self.buffers.len())
            .finish()
    }
}

impl ArrayParts {
    /// Creates a new [`ArrayParts`] from an [`OwnedArray`] flatbuffer.
    pub fn new(row_count: usize, array: OwnedArray, buffers: Vec<ByteBuffer>) -> Self {
        Self {
            row_count,
            array,
            buffers,
        }
    }

    /// Decode an [`ArrayParts`] into an [`ArrayData`].
    pub fn decode(self, ctx: ContextRef, dtype: DType) -> VortexResult<ArrayData> {
        ArrayData::try_new_viewed(ctx, dtype, self.row_count, self.array, self.buffers)
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

        // SAFETY: creating using the output of write_flatbuffer_bytes means we should have valid data.
        let owned_array = unsafe { OwnedArray::new_unchecked(flatbuffer) };

        let mut buffers: Vec<ByteBuffer> = vec![];
        for child in array.depth_first_traversal() {
            for buffer in child.byte_buffers() {
                buffers.push(buffer);
            }
        }

        Self {
            row_count: array.len(),
            array: owned_array,
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

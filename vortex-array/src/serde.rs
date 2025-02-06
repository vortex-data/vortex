use std::fmt::{Debug, Formatter};
use std::iter;

use flatbuffers::{root, FlatBufferBuilder, Follow, WIPOffset};
use itertools::Itertools;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_dtype::{DType, TryFromBytes};
use vortex_error::{vortex_bail, VortexError, VortexExpect, VortexResult};
use vortex_flatbuffers::array::Compression;
use vortex_flatbuffers::{array as fba, FlatBuffer, FlatBufferRoot, WriteFlatBuffer};

use crate::{Array, ContextRef};

/// Options for serializing an array.
#[derive(Default, Debug)]
pub struct SerializeOptions {
    /// The starting position within an external stream or file. This offset is used to compute
    /// appropriate padding to enable zero-copy reads.
    pub offset: usize,
    /// Whether to include sufficient zero-copy padding.
    pub include_padding: bool,
}

impl Array {
    /// Serialize the array into a sequence of byte buffers that should be written contiguously.
    /// This function returns a vec to avoid copying data buffers.
    ///
    /// Optionally, padding can be included to guarantee buffer alignment and ensure zero-copy
    /// reads within the context of an external file or stream. In this case, the alignment of
    /// the first byte buffer should be respected when writing the buffers to the stream or file.
    ///
    /// The format of this blob is a sequence of data buffers, possible with prefixed padding,
    /// followed by a flatbuffer containing an [`fba::Array`] message, and ending with a
    /// little-endian u32 describing the length of the flatbuffer message.
    pub fn serialize(&self, options: &SerializeOptions) -> Vec<ByteBuffer> {
        // Collect all array buffers
        let mut array_buffers = vec![];
        for a in self.depth_first_traversal() {
            for buffer in a.byte_buffers() {
                array_buffers.push(buffer);
            }
        }

        // Allocate result buffers, including a possible padding buffer for each.
        let mut buffers = vec![];
        let mut fb_buffers = Vec::with_capacity(buffers.capacity());

        // If we're including padding, we need to find the maximum required buffer alignment.
        let max_alignment = array_buffers
            .iter()
            .map(|buf| buf.alignment())
            .chain(iter::once(FlatBuffer::alignment()))
            .max()
            .vortex_expect("There is at least one alignment, the flatbuffer one");

        // Create a shared buffer of zeros we can use for padding
        let zeros = ByteBuffer::zeroed(*max_alignment);

        // We push an empty buffer with the maximum alignment, so then subsequent buffers
        // will be aligned. For subsequent buffers, we always push a 1-byte alignment.
        buffers.push(ByteBuffer::zeroed_aligned(0, max_alignment));

        // Keep track of where we are in the "file" to calculate padding.
        let mut pos = options.offset;

        // Push all the array buffers with padding as necessary.
        for buffer in array_buffers {
            let padding = if options.include_padding {
                let padding = pos.next_multiple_of(*buffer.alignment()) - pos;
                if padding > 0 {
                    pos += padding;
                    buffers.push(zeros.slice(0..padding));
                }
                padding
            } else {
                0
            };

            fb_buffers.push(fba::Buffer::new(
                u16::try_from(padding).vortex_expect("padding fits into u16"),
                buffer.alignment().exponent(),
                Compression::None,
                u32::try_from(buffer.len()).vortex_expect("buffers fit into u32"),
            ));
            pos += buffer.len();
            buffers.push(buffer.aligned(Alignment::none()));
        }

        // Set up the flatbuffer builder
        let mut fbb = FlatBufferBuilder::new();
        let root = ArrayNodeFlatBuffer::new(self);
        let fb_root = root.write_flatbuffer(&mut fbb);
        let fb_buffers = fbb.create_vector(&fb_buffers);
        let fb_array = fba::Array::create(
            &mut fbb,
            &fba::ArrayArgs {
                root: Some(fb_root),
                buffers: Some(fb_buffers),
            },
        );
        fbb.finish_minimal(fb_array);
        let (fb_vec, fb_start) = fbb.collapse();
        let fb_end = fb_vec.len();
        let fb_buffer = ByteBuffer::from(fb_vec).slice(fb_start..fb_end);
        let fb_length = fb_buffer.len();

        if options.include_padding {
            let padding = pos.next_multiple_of(*fb_buffer.alignment()) - pos;
            if padding > 0 {
                buffers.push(zeros.slice(0..padding));
            }
        }
        buffers.push(fb_buffer);

        // Finally, we write down the u32 length for the flatbuffer.
        buffers.push(ByteBuffer::from(
            u32::try_from(fb_length)
                .vortex_expect("u32 fits into usize")
                .to_le_bytes()
                .to_vec(),
        ));

        buffers
    }

    /// Deserialize an array from a [`ByteBuffer`].
    pub fn deserialize(
        bytes: ByteBuffer,
        ctx: ContextRef,
        dtype: DType,
        length: usize,
    ) -> VortexResult<Self> {
        ArrayParts::try_from(bytes)?.decode(ctx, dtype, length)
    }
}

/// A utility struct for creating an [`fba::ArrayNode`] flatbuffer.
pub struct ArrayNodeFlatBuffer<'a> {
    array: &'a Array,
    buffer_idx: u16,
}

impl<'a> ArrayNodeFlatBuffer<'a> {
    pub fn new(array: &'a Array) -> Self {
        Self {
            array,
            buffer_idx: 0,
        }
    }
}

impl FlatBufferRoot for ArrayNodeFlatBuffer<'_> {}

impl WriteFlatBuffer for ArrayNodeFlatBuffer<'_> {
    type Target<'t> = fba::ArrayNode<'t>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let encoding = self.array.encoding().code();
        let metadata = self
            .array
            .metadata_bytes()
            .map(|bytes| fbb.create_vector(bytes));

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
                let msg = ArrayNodeFlatBuffer {
                    array: child,
                    buffer_idx: *buffer_idx,
                }
                .write_flatbuffer(fbb);
                *buffer_idx = u16::try_from(child.cumulative_nbuffers())
                    .ok()
                    .and_then(|nbuffers| nbuffers.checked_add(*buffer_idx))
                    .vortex_expect("Too many buffers (u16) for Array");
                Some(msg)
            })
            .collect_vec();
        let children = Some(fbb.create_vector(&children));

        let buffers = Some(fbb.create_vector_from_iter((0..nbuffers).map(|i| i + self.buffer_idx)));
        let stats = Some(self.array.statistics().write_flatbuffer(fbb));

        fba::ArrayNode::create(
            fbb,
            &fba::ArrayNodeArgs {
                encoding,
                metadata,
                children,
                buffers,
                stats,
            },
        )
    }
}

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

impl TryFrom<ByteBuffer> for ArrayParts {
    type Error = VortexError;

    fn try_from(value: ByteBuffer) -> Result<Self, Self::Error> {
        // The final 4 bytes contain the length of the flatbuffer.
        if value.len() < 4 {
            vortex_bail!("ArrayParts buffer is too short");
        }

        // We align each buffer individually, so we remove alignment requirements on the buffer.
        let value = value.aligned(Alignment::none());

        let fb_length = u32::try_from_le_bytes(&value.as_slice()[value.len() - 4..])? as usize;
        let fb_offset = value.len() - 4 - fb_length;
        let fb_buffer = FlatBuffer::align_from(value.slice(fb_offset..fb_offset + fb_length));

        let fb_array = root::<fba::Array>(fb_buffer.as_ref())?;
        let fb_root = fb_array.root().vortex_expect("Array must have a root node");

        let mut offset = 0;
        let buffers = fb_array
            .buffers()
            .unwrap_or_default()
            .iter()
            .map(|fb_buffer| {
                // Skip padding
                offset += fb_buffer.padding() as usize;

                let buffer_len = fb_buffer.length() as usize;

                // Extract a buffer and ensure it's aligned, copying if necessary
                let buffer = value
                    .slice(offset..(offset + buffer_len))
                    .aligned(Alignment::from_exponent(fb_buffer.alignment_exponent()));

                offset += buffer_len;
                buffer
            })
            .collect_vec();

        Ok(ArrayParts {
            flatbuffer: fb_buffer.clone(),
            flatbuffer_loc: fb_root._tab.loc(),
            buffers,
        })
    }
}

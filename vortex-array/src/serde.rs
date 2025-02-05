use std::iter;

use flatbuffers::{FlatBufferBuilder, WIPOffset};
use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_flatbuffers::array::Compression;
use vortex_flatbuffers::{array as fba, FlatBuffer, FlatBufferRoot, WriteFlatBuffer};

use crate::Array;

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
    /// reads within the context of an external file or stream.
    ///
    /// The format of this blob is a sequence of data buffers, possible with prefixed padding,
    /// followed by a flatbuffer containing an [`fba::Array`] message, and ending with a
    /// little-endian u32 describing the length of the flatbuffer message.
    ///
    /// TODO(ngates): by this point we should return `bytes::Bytes` since we no longer care for
    ///  alignment
    pub fn serialize(&self, options: &SerializeOptions) -> Vec<ByteBuffer> {
        // Collect all array buffers
        let mut array_buffers = vec![];
        for a in self.depth_first_traversal() {
            for buffer in a.byte_buffers() {
                array_buffers.push(buffer);
            }
        }

        // Allocate result buffers, including a possible padding buffer for each.
        let mut buffers = Vec::with_capacity(2 * (array_buffers.len() + 1));

        // Set up the flatbuffer builder
        let mut fbb = FlatBufferBuilder::new();
        let mut fb_buffers = Vec::with_capacity(buffers.capacity());

        // If we're including padding, we need to find the maximum required buffer alignment.
        let alignment = array_buffers
            .iter()
            .map(|buf| buf.alignment())
            .chain(iter::once(FlatBuffer::alignment()))
            .max()
            .vortex_expect("There is at least one alignment, the flatbuffer one");

        // Create a shared buffer of zeros we can use for padding
        let zeros = ByteBuffer::zeroed(*alignment);

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
            buffers.push(buffer);
        }

        let fb_root = ArrayNodeFlatBuffer::new(&self).write_flatbuffer(&mut fbb);
        let fb_buffers = fbb.create_vector(&fb_buffers);
        let fb_array = fba::Array::create(
            &mut fbb,
            &fba::ArrayArgs {
                root: Some(fb_root),
                buffers: Some(fb_buffers),
            },
        );
        fbb.finish_minimal(fb_array);
        let (fb_vec, fb_offset) = fbb.collapse();
        let fb_buffer =
            ByteBuffer::copy_from_aligned(&fb_vec[fb_offset..], FlatBuffer::alignment());

        if options.include_padding {
            let padding = pos.next_multiple_of(*fb_buffer.alignment()) - pos;
            if padding > 0 {
                pos += padding;
                buffers.push(zeros.slice(0..padding));
            }
        }
        buffers.push(fb_buffer);

        // Finally, we write down the u32 length for the flatbuffer.
        buffers.push(ByteBuffer::from(
            u32::try_from(pos)
                .vortex_expect("u32 fits into usize")
                .to_le_bytes()
                .to_vec(),
        ));

        buffers
    }

    /// Deserialize an array from a [`ByteBuffer`].
    pub fn deserialize(_bytes: ByteBuffer) -> Self {
        todo!()
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

use std::fmt::{Debug, Formatter};
use std::iter;
use std::sync::Arc;

use flatbuffers::{FlatBufferBuilder, Follow, WIPOffset, root};
use itertools::Itertools;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_dtype::{DType, TryFromBytes};
use vortex_error::{
    VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err, vortex_panic,
};
use vortex_flatbuffers::array::Compression;
use vortex_flatbuffers::{
    FlatBuffer, FlatBufferRoot, ReadFlatBuffer, WriteFlatBuffer, array as fba,
};

use crate::stats::StatsSet;
use crate::{Array, ArrayRef, ArrayVisitor, ArrayVisitorExt, ContextRef};

/// Options for serializing an array.
#[derive(Default, Debug)]
pub struct SerializeOptions {
    /// The starting position within an external stream or file. This offset is used to compute
    /// appropriate padding to enable zero-copy reads.
    pub offset: usize,
    /// Whether to include sufficient zero-copy padding.
    pub include_padding: bool,
}

impl dyn Array + '_ {
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
            for buffer in a.buffers() {
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
            .unwrap_or_else(FlatBuffer::alignment);

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
            let padding = pos.next_multiple_of(*FlatBuffer::alignment()) - pos;
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
}

/// A utility struct for creating an [`fba::ArrayNode`] flatbuffer.
pub struct ArrayNodeFlatBuffer<'a> {
    array: &'a dyn Array,
    buffer_idx: u16,
}

impl<'a> ArrayNodeFlatBuffer<'a> {
    pub fn new(array: &'a dyn Array) -> Self {
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
            .metadata()
            .map(|bytes| fbb.create_vector(bytes.as_slice()));

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
                *buffer_idx = u16::try_from(child.nbuffers_recursive())
                    .ok()
                    .and_then(|nbuffers| nbuffers.checked_add(*buffer_idx))
                    .vortex_expect("Too many buffers (u16) for Array");
                Some(msg)
            })
            .collect_vec();
        let children = Some(fbb.create_vector(&children));

        let buffers = Some(fbb.create_vector_from_iter((0..nbuffers).map(|i| i + self.buffer_idx)));
        let stats = Some(self.array.statistics().stats_set().write_flatbuffer(fbb));

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

/// [`ArrayParts`] represents the information from an [`ArrayRef`] that makes up the serialized
/// form. For example, it uses stores integer encoding IDs rather than a reference to an encoding
/// vtable, and it doesn't store any [`DType`] information.
///
/// An [`ArrayParts`] can be fully decoded into an [`ArrayRef`] using the `decode` function.
#[derive(Clone)]
pub struct ArrayParts {
    // Typed as fb::ArrayNode
    flatbuffer: FlatBuffer,
    // The location of the current fb::ArrayNode
    flatbuffer_loc: usize,
    buffers: Arc<[ByteBuffer]>,
}

impl Debug for ArrayParts {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayParts")
            .field("encoding_id", &self.encoding_id())
            .field("children", &(0..self.nchildren()).map(|i| self.child(i)))
            .field(
                "buffers",
                &(0..self.nbuffers()).map(|i| self.buffer(i).ok()),
            )
            .field("metadata", &self.metadata())
            .finish()
    }
}

impl ArrayParts {
    /// Decode an [`ArrayParts`] into an [`ArrayRef`].
    pub fn decode(&self, ctx: &ContextRef, dtype: DType, len: usize) -> VortexResult<ArrayRef> {
        let encoding_id = self.flatbuffer().encoding();
        let vtable = ctx
            .lookup_encoding(encoding_id)
            .ok_or_else(|| vortex_err!("Unknown encoding: {}", encoding_id))?;
        let decoded = vtable.decode(self, ctx, dtype, len)?;
        assert_eq!(
            decoded.len(),
            len,
            "Array decoded from {} has incorrect length {}, expected {}",
            vtable.id(),
            decoded.len(),
            len
        );
        assert_eq!(
            decoded.encoding(),
            vtable.id(),
            "Array decoded from {} has incorrect encoding {}",
            vtable.id(),
            decoded.encoding(),
        );

        // Populate statistics from the serialized array.
        if let Some(stats) = self.flatbuffer().stats() {
            let decoded_statistics = decoded.statistics();
            StatsSet::read_flatbuffer(&stats)?
                .into_iter()
                .for_each(|(stat, val)| decoded_statistics.set_stat(stat, val));
        }

        Ok(decoded)
    }

    /// Returns the array encoding.
    pub fn encoding_id(&self) -> u16 {
        self.flatbuffer().encoding()
    }

    /// Returns the array metadata bytes.
    pub fn metadata(&self) -> Option<&[u8]> {
        self.flatbuffer()
            .metadata()
            .map(|metadata| metadata.bytes())
    }

    /// Returns the number of children.
    pub fn nchildren(&self) -> usize {
        self.flatbuffer()
            .children()
            .map_or(0, |children| children.len())
    }

    /// Returns the nth child of the array.
    pub fn child(&self, idx: usize) -> ArrayParts {
        let children = self
            .flatbuffer()
            .children()
            .vortex_expect("Expected array to have children");
        if idx >= children.len() {
            vortex_panic!(
                "Invalid child index {} for array with {} children",
                idx,
                children.len()
            );
        }
        self.with_root(children.get(idx))
    }

    /// Iterate the children of this array.
    pub fn children(&self) -> Vec<ArrayParts> {
        self.flatbuffer()
            .children()
            .iter()
            .flat_map(|children| children.iter())
            .map(move |child| self.with_root(child))
            .collect()
    }

    /// Returns the number of buffers.
    pub fn nbuffers(&self) -> usize {
        self.flatbuffer()
            .buffers()
            .map_or(0, |buffers| buffers.len())
    }

    /// Returns the nth buffer of the current array.
    pub fn buffer(&self, idx: usize) -> VortexResult<ByteBuffer> {
        let buffer_idx = self
            .flatbuffer()
            .buffers()
            .ok_or_else(|| vortex_err!("Array has no buffers"))?
            .get(idx);
        self.buffers
            .get(buffer_idx as usize)
            .cloned()
            .ok_or_else(|| {
                vortex_err!(
                    "Invalid buffer index {} for array with {} buffers",
                    buffer_idx,
                    self.nbuffers()
                )
            })
    }

    /// Returns the root ArrayNode flatbuffer.
    fn flatbuffer(&self) -> fba::ArrayNode {
        unsafe { fba::ArrayNode::follow(self.flatbuffer.as_ref(), self.flatbuffer_loc) }
    }

    /// Returns a new [`ArrayParts`] with the given node as the root
    // TODO(ngates): we may want a wrapper that avoids this clone.
    fn with_root(&self, root: fba::ArrayNode) -> Self {
        let mut this = self.clone();
        this.flatbuffer_loc = root._tab.loc();
        this
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
        let fb_buffer = value.slice(fb_offset..fb_offset + fb_length);
        let fb_buffer = FlatBuffer::align_from(fb_buffer);

        let fb_array = root::<fba::Array>(fb_buffer.as_ref())?;
        let fb_root = fb_array.root().vortex_expect("Array must have a root node");

        let mut offset = 0;
        let buffers: Arc<[ByteBuffer]> = fb_array
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
            .collect();

        Ok(ArrayParts {
            flatbuffer: fb_buffer.clone(),
            flatbuffer_loc: fb_root._tab.loc(),
            buffers,
        })
    }
}

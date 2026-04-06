// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::iter;
use std::sync::Arc;

use flatbuffers::FlatBufferBuilder;
use flatbuffers::Follow;
use flatbuffers::WIPOffset;
use flatbuffers::root;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::array as fba;
use vortex_flatbuffers::array::Compression;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayContext;
use crate::ArrayRef;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::TryFromBytes;
use crate::session::ArraySessionExt;
use crate::stats::StatsSet;

/// Options for serializing an array.
#[derive(Default, Debug)]
pub struct SerializeOptions {
    /// The starting position within an external stream or file. This offset is used to compute
    /// appropriate padding to enable zero-copy reads.
    pub offset: usize,
    /// Whether to include sufficient zero-copy padding.
    pub include_padding: bool,
}

impl ArrayRef {
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
    pub fn serialize(
        &self,
        ctx: &ArrayContext,
        options: &SerializeOptions,
    ) -> VortexResult<Vec<ByteBuffer>> {
        // Collect all array buffers
        let array_buffers = self
            .depth_first_traversal()
            .flat_map(|f| f.buffers())
            .collect::<Vec<_>>();

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
                u32::try_from(buffer.len())
                    .map_err(|_| vortex_err!("All buffers must fit into u32 for serialization"))?,
            ));

            pos += buffer.len();
            buffers.push(buffer.aligned(Alignment::none()));
        }

        // Set up the flatbuffer builder
        let mut fbb = FlatBufferBuilder::new();

        let root = ArrayNodeFlatBuffer::try_new(ctx, self)?;
        let fb_root = root.try_write_flatbuffer(&mut fbb)?;

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
                .map_err(|_| vortex_err!("Array metadata flatbuffer must fit into u32 for serialization. Array encoding tree is too large."))?
                .to_le_bytes()
                .to_vec(),
        ));

        Ok(buffers)
    }
}

/// A utility struct for creating an [`fba::ArrayNode`] flatbuffer.
pub struct ArrayNodeFlatBuffer<'a> {
    ctx: &'a ArrayContext,
    array: &'a ArrayRef,
    buffer_idx: u16,
}

impl<'a> ArrayNodeFlatBuffer<'a> {
    pub fn try_new(ctx: &'a ArrayContext, array: &'a ArrayRef) -> VortexResult<Self> {
        // Depth-first traversal of the array to ensure it supports serialization.
        for child in array.depth_first_traversal() {
            if child.metadata()?.is_none() {
                vortex_bail!(
                    "Array {} does not support serialization",
                    child.encoding_id()
                );
            }
        }
        let n_buffers_recursive = array.nbuffers_recursive();
        if n_buffers_recursive > u16::MAX as usize {
            vortex_bail!(
                "Array and all descendent arrays can have at most u16::MAX buffers: {}",
                n_buffers_recursive
            );
        };
        Ok(Self {
            ctx,
            array,
            buffer_idx: 0,
        })
    }

    pub fn try_write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<fba::ArrayNode<'fb>>> {
        let encoding_idx = self
            .ctx
            .intern(&self.array.encoding_id())
            // TODO(ngates): write_flatbuffer should return a result if this can fail.
            .ok_or_else(|| {
                vortex_err!(
                    "Array encoding {} not permitted by ctx",
                    self.array.encoding_id()
                )
            })?;

        let metadata = self.array.metadata()?.ok_or_else(|| {
            vortex_err!(
                "Array {} does not support serialization",
                self.array.encoding_id()
            )
        })?;
        let metadata = Some(fbb.create_vector(metadata.as_slice()));

        // Assign buffer indices for all child arrays.
        let nbuffers = u16::try_from(self.array.nbuffers())
            .map_err(|_| vortex_err!("Array can have at most u16::MAX buffers"))?;
        let mut child_buffer_idx = self.buffer_idx + nbuffers;

        let children = self
            .array
            .children()
            .iter()
            .map(|child| {
                // Update the number of buffers required.
                let msg = ArrayNodeFlatBuffer {
                    ctx: self.ctx,
                    array: child,
                    buffer_idx: child_buffer_idx,
                }
                .try_write_flatbuffer(fbb)?;

                child_buffer_idx = u16::try_from(child.nbuffers_recursive())
                    .ok()
                    .and_then(|nbuffers| nbuffers.checked_add(child_buffer_idx))
                    .ok_or_else(|| vortex_err!("Too many buffers (u16) for Array"))?;

                Ok(msg)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let children = Some(fbb.create_vector(&children));

        let buffers = Some(fbb.create_vector_from_iter((0..nbuffers).map(|i| i + self.buffer_idx)));
        let stats = Some(self.array.statistics().write_flatbuffer(fbb)?);

        Ok(fba::ArrayNode::create(
            fbb,
            &fba::ArrayNodeArgs {
                encoding: encoding_idx,
                metadata,
                children,
                buffers,
                stats,
            },
        ))
    }
}

/// To minimize the serialized form, arrays do not persist their own dtype and length. Instead,
/// parent arrays pass this information down during deserialization.
pub trait ArrayChildren {
    /// Returns the nth child of the array with the given dtype and length.
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef>;

    /// The number of children.
    fn len(&self) -> usize;

    /// Returns true if there are no children.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ArrayChildren for &[ArrayRef] {
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
        let array = self[index].clone();
        assert_eq!(array.len(), len);
        assert_eq!(array.dtype(), dtype);
        Ok(array)
    }

    fn len(&self) -> usize {
        <[_]>::len(self)
    }
}

/// [`ArrayParts`] represents a parsed but not-yet-decoded deserialized array.
/// It contains all the information from the serialized form, without anything extra. i.e.
/// it is missing a [`DType`] and `len`, and the `encoding_id` is not yet resolved to a concrete
/// vtable.
///
/// An [`ArrayParts`] can be fully decoded into an [`ArrayRef`] using the `decode` function.
#[derive(Clone)]
pub struct ArrayParts {
    // Typed as fb::ArrayNode
    flatbuffer: FlatBuffer,
    // The location of the current fb::ArrayNode
    flatbuffer_loc: usize,
    buffers: Arc<[BufferHandle]>,
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
    pub fn decode(
        &self,
        dtype: &DType,
        len: usize,
        ctx: &ReadContext,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let encoding_idx = self.flatbuffer().encoding();
        let encoding_id = ctx
            .resolve(encoding_idx)
            .ok_or_else(|| vortex_err!("Unknown encoding index: {}", encoding_idx))?;
        let vtable = session
            .arrays()
            .registry()
            .find(&encoding_id)
            .ok_or_else(|| vortex_err!("Unknown encoding: {}", encoding_id))?;

        let children = ArrayPartsChildren {
            parts: self,
            ctx,
            session,
        };

        let buffers = self.collect_buffers()?;

        let decoded = vtable.build(
            encoding_id.clone(),
            dtype,
            len,
            self.metadata(),
            &buffers,
            &children,
            session,
        )?;

        assert_eq!(
            decoded.len(),
            len,
            "Array decoded from {} has incorrect length {}, expected {}",
            encoding_id,
            decoded.len(),
            len
        );
        assert_eq!(
            decoded.dtype(),
            dtype,
            "Array decoded from {} has incorrect dtype {}, expected {}",
            encoding_id,
            decoded.dtype(),
            dtype,
        );
        assert_eq!(
            decoded.encoding_id(),
            encoding_id,
            "Array decoded from {} has incorrect encoding {}",
            encoding_id,
            decoded.encoding_id(),
        );

        // Populate statistics from the serialized array.
        if let Some(stats) = self.flatbuffer().stats() {
            decoded
                .statistics()
                .set_iter(StatsSet::from_flatbuffer(&stats, dtype, session)?.into_iter());
        }

        Ok(decoded)
    }

    /// Returns the array encoding.
    pub fn encoding_id(&self) -> u16 {
        self.flatbuffer().encoding()
    }

    /// Returns the array metadata bytes.
    pub fn metadata(&self) -> &[u8] {
        self.flatbuffer()
            .metadata()
            .map(|metadata| metadata.bytes())
            .unwrap_or(&[])
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

    /// Returns the number of buffers.
    pub fn nbuffers(&self) -> usize {
        self.flatbuffer()
            .buffers()
            .map_or(0, |buffers| buffers.len())
    }

    /// Returns the nth buffer of the current array.
    pub fn buffer(&self, idx: usize) -> VortexResult<BufferHandle> {
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

    /// Returns all buffers for the current array node.
    ///
    /// If buffer indices are contiguous, returns a zero-copy borrowed slice.
    /// Otherwise falls back to collecting each buffer individually.
    fn collect_buffers(&self) -> VortexResult<Cow<'_, [BufferHandle]>> {
        let Some(fb_buffers) = self.flatbuffer().buffers() else {
            return Ok(Cow::Borrowed(&[]));
        };
        let count = fb_buffers.len();
        if count == 0 {
            return Ok(Cow::Borrowed(&[]));
        }
        let start = fb_buffers.get(0) as usize;
        let contiguous = fb_buffers
            .iter()
            .enumerate()
            .all(|(i, idx)| idx as usize == start + i);
        if contiguous {
            self.buffers.get(start..start + count).map_or_else(
                || {
                    vortex_bail!(
                        "buffer indices {}..{} out of range for {} buffers",
                        start,
                        start + count,
                        self.buffers.len()
                    )
                },
                |slice| Ok(Cow::Borrowed(slice)),
            )
        } else {
            (0..count)
                .map(|idx| self.buffer(idx))
                .collect::<VortexResult<Vec<_>>>()
                .map(Cow::Owned)
        }
    }

    /// Returns the buffer lengths as stored in the flatbuffer metadata.
    ///
    /// This reads the buffer descriptors from the flatbuffer, which contain the
    /// serialized length of each buffer. This is useful for displaying buffer sizes
    /// without needing to access the actual buffer data.
    pub fn buffer_lengths(&self) -> Vec<usize> {
        let fb_array = root::<fba::Array>(self.flatbuffer.as_ref())
            .vortex_expect("ArrayParts flatbuffer must be a valid Array");
        fb_array
            .buffers()
            .map(|buffers| buffers.iter().map(|b| b.length() as usize).collect())
            .unwrap_or_default()
    }

    /// Validate and align the array tree flatbuffer, returning the aligned buffer and root location.
    fn validate_array_tree(array_tree: impl Into<ByteBuffer>) -> VortexResult<(FlatBuffer, usize)> {
        let fb_buffer = FlatBuffer::align_from(array_tree.into());
        let fb_array = root::<fba::Array>(fb_buffer.as_ref())?;
        let fb_root = fb_array
            .root()
            .ok_or_else(|| vortex_err!("Array must have a root node"))?;
        let flatbuffer_loc = fb_root._tab.loc();
        Ok((fb_buffer, flatbuffer_loc))
    }

    /// Create an [`ArrayParts`] from a pre-existing array tree flatbuffer and pre-resolved buffer
    /// handles.
    ///
    /// The caller is responsible for resolving buffers from whatever source (device segments, host
    /// overrides, or a mix). The buffers must be in the same order as the `Array.buffers` descriptor
    /// list in the flatbuffer.
    pub fn from_flatbuffer_with_buffers(
        array_tree: impl Into<ByteBuffer>,
        buffers: Vec<BufferHandle>,
    ) -> VortexResult<Self> {
        let (flatbuffer, flatbuffer_loc) = Self::validate_array_tree(array_tree)?;
        Ok(ArrayParts {
            flatbuffer,
            flatbuffer_loc,
            buffers: buffers.into(),
        })
    }

    /// Create an [`ArrayParts`] from a raw array tree flatbuffer (metadata only).
    ///
    /// This constructor creates an `ArrayParts` with no buffer data, useful for
    /// inspecting the metadata when the actual buffer data is not needed
    /// (e.g., displaying buffer sizes from inlined array tree metadata).
    ///
    /// Note: Calling `buffer()` on the returned `ArrayParts` will fail since
    /// no actual buffer data is available.
    pub fn from_array_tree(array_tree: impl Into<ByteBuffer>) -> VortexResult<Self> {
        let (flatbuffer, flatbuffer_loc) = Self::validate_array_tree(array_tree)?;
        Ok(ArrayParts {
            flatbuffer,
            flatbuffer_loc,
            buffers: Arc::new([]),
        })
    }

    /// Returns the root ArrayNode flatbuffer.
    fn flatbuffer(&self) -> fba::ArrayNode<'_> {
        unsafe { fba::ArrayNode::follow(self.flatbuffer.as_ref(), self.flatbuffer_loc) }
    }

    /// Returns a new [`ArrayParts`] with the given node as the root
    // TODO(ngates): we may want a wrapper that avoids this clone.
    fn with_root(&self, root: fba::ArrayNode) -> Self {
        let mut this = self.clone();
        this.flatbuffer_loc = root._tab.loc();
        this
    }

    /// Create an [`ArrayParts`] from a pre-existing flatbuffer (ArrayNode) and a segment containing
    /// only the data buffers (without the flatbuffer suffix).
    ///
    /// This is used when the flatbuffer is stored separately in layout metadata (e.g., when
    /// `FLAT_LAYOUT_INLINE_ARRAY_NODE` is enabled).
    pub fn from_flatbuffer_and_segment(
        array_tree: ByteBuffer,
        segment: BufferHandle,
    ) -> VortexResult<Self> {
        // HashMap::new doesn't allocate when empty, so this has no overhead
        Self::from_flatbuffer_and_segment_with_overrides(array_tree, segment, &HashMap::new())
    }

    /// Create an [`ArrayParts`] from a pre-existing flatbuffer (ArrayNode) and a segment,
    /// substituting host-resident buffer overrides for specific buffer indices.
    ///
    /// Buffers whose index appears in `buffer_overrides` are resolved from the provided
    /// host data instead of the segment. All other buffers are sliced from the segment
    /// using the padding and alignment described in the flatbuffer.
    pub fn from_flatbuffer_and_segment_with_overrides(
        array_tree: ByteBuffer,
        segment: BufferHandle,
        buffer_overrides: &HashMap<u32, ByteBuffer>,
    ) -> VortexResult<Self> {
        // We align each buffer individually, so we remove alignment requirements on the segment
        // for host-resident buffers. Device buffers are sliced directly.
        let segment = segment.ensure_aligned(Alignment::none())?;

        // this can't return the validated array because there is no lifetime to give it, so we
        // need to cast it below, which is safe.
        let (fb_buffer, flatbuffer_loc) = Self::validate_array_tree(array_tree)?;
        // SAFETY: fb_buffer was already validated by validate_array_tree above.
        let fb_array = unsafe { fba::root_as_array_unchecked(fb_buffer.as_ref()) };

        let mut offset = 0;
        let buffers = fb_array
            .buffers()
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(idx, fb_buf)| {
                offset += fb_buf.padding() as usize;
                let buffer_len = fb_buf.length() as usize;
                let alignment = Alignment::from_exponent(fb_buf.alignment_exponent());

                let idx = u32::try_from(idx).vortex_expect("buffer count must fit in u32");
                let handle = if let Some(host_data) = buffer_overrides.get(&idx) {
                    BufferHandle::new_host(host_data.clone()).ensure_aligned(alignment)?
                } else {
                    let buffer = segment.slice(offset..(offset + buffer_len));
                    buffer.ensure_aligned(alignment)?
                };

                offset += buffer_len;
                Ok(handle)
            })
            .collect::<VortexResult<Arc<[_]>>>()?;

        Ok(ArrayParts {
            flatbuffer: fb_buffer,
            flatbuffer_loc,
            buffers,
        })
    }
}

struct ArrayPartsChildren<'a> {
    parts: &'a ArrayParts,
    ctx: &'a ReadContext,
    session: &'a VortexSession,
}

impl ArrayChildren for ArrayPartsChildren<'_> {
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
        self.parts
            .child(index)
            .decode(dtype, len, self.ctx, self.session)
    }

    fn len(&self) -> usize {
        self.parts.nchildren()
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
        if value.len() < 4 + fb_length {
            vortex_bail!("ArrayParts buffer is too short for flatbuffer");
        }

        let fb_offset = value.len() - 4 - fb_length;
        let array_tree = value.slice(fb_offset..fb_offset + fb_length);
        let segment = BufferHandle::new_host(value.slice(0..fb_offset));

        Self::from_flatbuffer_and_segment(array_tree, segment)
    }
}

impl TryFrom<BufferHandle> for ArrayParts {
    type Error = VortexError;

    fn try_from(value: BufferHandle) -> Result<Self, Self::Error> {
        Self::try_from(value.try_to_host_sync()?)
    }
}

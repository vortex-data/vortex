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
use crate::ArraySlots;
use crate::array::ArrayId;
use crate::array::new_foreign_array;
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

/// Collect flatbuffer buffer descriptors from array buffers, computing padding for each.
///
/// This is the shared logic between [`ArrayRef::serialize`] and [`ArrayRef::serialize_array_tree`]
/// to ensure buffer descriptor tables are always consistent.
fn collect_buffer_descriptors(
    array_buffers: &[ByteBuffer],
    options: &SerializeOptions,
) -> VortexResult<Vec<fba::Buffer>> {
    let mut fb_buffers = Vec::with_capacity(array_buffers.len());
    let mut pos = options.offset;

    for buffer in array_buffers {
        let padding = if options.include_padding {
            let padding = pos.next_multiple_of(*buffer.alignment()) - pos;
            pos += padding;
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
    }

    Ok(fb_buffers)
}

/// Build a complete `fba::Array` flatbuffer from an encoding tree and buffer descriptors.
fn build_array_flatbuffer(
    ctx: &ArrayContext,
    session: &VortexSession,
    array: &ArrayRef,
    fb_buffers: Vec<fba::Buffer>,
    skip_stats: bool,
) -> VortexResult<ByteBuffer> {
    let mut fbb = FlatBufferBuilder::new();

    let mut root = ArrayNodeFlatBuffer::try_new(ctx, session, array)?;
    root.skip_stats = skip_stats;
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
    Ok(ByteBuffer::from(fb_vec).slice(fb_start..fb_end))
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
        session: &VortexSession,
        options: &SerializeOptions,
    ) -> VortexResult<Vec<ByteBuffer>> {
        // Collect all array buffers
        let array_buffers = self
            .depth_first_traversal()
            .flat_map(|f| f.buffers())
            .collect::<Vec<_>>();

        let fb_buffers = collect_buffer_descriptors(&array_buffers, options)?;

        // Allocate result buffers, including a possible padding buffer for each.
        let mut buffers = vec![];

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
            if options.include_padding {
                let padding = pos.next_multiple_of(*buffer.alignment()) - pos;
                if padding > 0 {
                    pos += padding;
                    buffers.push(zeros.slice(0..padding));
                }
            }

            pos += buffer.len();
            buffers.push(buffer.aligned(Alignment::none()));
        }

        let fb_buffer = build_array_flatbuffer(ctx, session, self, fb_buffers, false)?;
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

    /// Produce a compact [`fba::Array`] flatbuffer containing the encoding tree and buffer
    /// descriptors, but with per-node statistics stripped (`stats = null` on all [`fba::ArrayNode`]s).
    ///
    /// This is used by the array tree layout to store encoding metadata separately from data
    /// segments, enabling decode planning and sub-segment random access without fetching
    /// the full data segment.
    ///
    /// The returned flatbuffer has the same `buffers` table as a full [`serialize`](Self::serialize)
    /// call with the same options, so buffer offsets can be used for sub-segment reads.
    pub fn serialize_array_tree(
        &self,
        ctx: &ArrayContext,
        session: &VortexSession,
        options: &SerializeOptions,
    ) -> VortexResult<ByteBuffer> {
        let array_buffers = self
            .depth_first_traversal()
            .flat_map(|f| f.buffers())
            .collect::<Vec<_>>();

        let fb_buffers = collect_buffer_descriptors(&array_buffers, options)?;
        build_array_flatbuffer(ctx, session, self, fb_buffers, true)
    }
}

/// A utility struct for creating an [`fba::ArrayNode`] flatbuffer.
pub struct ArrayNodeFlatBuffer<'a> {
    ctx: &'a ArrayContext,
    session: &'a VortexSession,
    array: &'a ArrayRef,
    buffer_idx: u16,
    skip_stats: bool,
}

impl<'a> ArrayNodeFlatBuffer<'a> {
    pub fn try_new(
        ctx: &'a ArrayContext,
        session: &'a VortexSession,
        array: &'a ArrayRef,
    ) -> VortexResult<Self> {
        let n_buffers_recursive = array.nbuffers_recursive();
        if n_buffers_recursive > u16::MAX as usize {
            vortex_bail!(
                "Array and all descendent arrays can have at most u16::MAX buffers: {}",
                n_buffers_recursive
            );
        };
        Ok(Self {
            ctx,
            session,
            array,
            buffer_idx: 0,
            skip_stats: false,
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

        let metadata_bytes = self.session.array_serialize(self.array)?.ok_or_else(|| {
            vortex_err!(
                "Array {} does not support serialization",
                self.array.encoding_id()
            )
        })?;
        let metadata = Some(fbb.create_vector(metadata_bytes.as_slice()));

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
                    session: self.session,
                    array: child,
                    buffer_idx: child_buffer_idx,
                    skip_stats: self.skip_stats,
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
        let stats = if self.skip_stats {
            None
        } else {
            Some(self.array.statistics().write_flatbuffer(fbb)?)
        };

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

impl<T: AsRef<[ArrayRef]>> ArrayChildren for T {
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
        let array = self.as_ref()[index].clone();
        assert_eq!(array.len(), len);
        assert_eq!(array.dtype(), dtype);
        Ok(array)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

/// [`SerializedArray`] represents a parsed but not-yet-decoded deserialized array.
/// It contains all the information from the serialized form, without anything extra. i.e.
/// it is missing a [`DType`] and `len`, and the `encoding_id` is not yet resolved to a concrete
/// vtable.
///
/// An [`SerializedArray`] can be fully decoded into an [`ArrayRef`] using the `decode` function.
#[derive(Clone)]
pub struct SerializedArray {
    // Typed as fb::ArrayNode
    flatbuffer: FlatBuffer,
    // The location of the current fb::ArrayNode
    flatbuffer_loc: usize,
    buffers: Arc<[BufferHandle]>,
}

impl Debug for SerializedArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SerializedArray")
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

impl SerializedArray {
    /// Decode an [`SerializedArray`] into an [`ArrayRef`].
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
        let Some(plugin) = session.arrays().registry().find(&encoding_id) else {
            if session.allows_unknown() {
                return self.decode_foreign(encoding_id, dtype, len, ctx);
            }
            return Err(vortex_err!("Unknown encoding: {}", encoding_id));
        };

        let children = SerializedArrayChildren {
            ser: self,
            ctx,
            session,
        };

        let buffers = self.collect_buffers()?;

        let decoded =
            plugin.deserialize(dtype, len, self.metadata(), &buffers, &children, session)?;

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

        assert!(
            plugin.is_supported_encoding(&decoded.encoding_id()),
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

    fn decode_foreign(
        &self,
        encoding_id: ArrayId,
        dtype: &DType,
        len: usize,
        ctx: &ReadContext,
    ) -> VortexResult<ArrayRef> {
        let children = (0..self.nchildren())
            .map(|idx| {
                let child = self.child(idx);
                let child_encoding_idx = child.flatbuffer().encoding();
                let child_encoding_id = ctx
                    .resolve(child_encoding_idx)
                    .ok_or_else(|| vortex_err!("Unknown encoding index: {}", child_encoding_idx))?;
                child
                    .decode_foreign(child_encoding_id, dtype, len, ctx)
                    .map(Some)
            })
            .collect::<VortexResult<ArraySlots>>()?;

        new_foreign_array(
            encoding_id,
            dtype.clone(),
            len,
            self.metadata().to_vec(),
            self.collect_buffers()?.into_owned(),
            children,
        )
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
    pub fn child(&self, idx: usize) -> SerializedArray {
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
            .vortex_expect("SerializedArray flatbuffer must be a valid Array");
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

    /// Create an [`SerializedArray`] from a pre-existing array tree flatbuffer and pre-resolved buffer
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
        Ok(SerializedArray {
            flatbuffer,
            flatbuffer_loc,
            buffers: buffers.into(),
        })
    }

    /// Create an [`SerializedArray`] from a raw array tree flatbuffer (metadata only).
    ///
    /// This constructor creates a `SerializedArray` with no buffer data, useful for
    /// inspecting the metadata when the actual buffer data is not needed
    /// (e.g., displaying buffer sizes from inlined array tree metadata).
    ///
    /// Note: Calling `buffer()` on the returned `SerializedArray` will fail since
    /// no actual buffer data is available.
    pub fn from_array_tree(array_tree: impl Into<ByteBuffer>) -> VortexResult<Self> {
        let (flatbuffer, flatbuffer_loc) = Self::validate_array_tree(array_tree)?;
        Ok(SerializedArray {
            flatbuffer,
            flatbuffer_loc,
            buffers: Arc::new([]),
        })
    }

    /// Returns the root ArrayNode flatbuffer.
    fn flatbuffer(&self) -> fba::ArrayNode<'_> {
        unsafe { fba::ArrayNode::follow(self.flatbuffer.as_ref(), self.flatbuffer_loc) }
    }

    /// Returns a new [`SerializedArray`] with the given node as the root
    // TODO(ngates): we may want a wrapper that avoids this clone.
    fn with_root(&self, root: fba::ArrayNode) -> Self {
        let mut this = self.clone();
        this.flatbuffer_loc = root._tab.loc();
        this
    }

    /// Create an [`SerializedArray`] from a pre-existing flatbuffer (ArrayNode) and a segment containing
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

    /// Create an [`SerializedArray`] from a pre-existing flatbuffer (ArrayNode) and a segment,
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

        Ok(SerializedArray {
            flatbuffer: fb_buffer,
            flatbuffer_loc,
            buffers,
        })
    }
}

struct SerializedArrayChildren<'a> {
    ser: &'a SerializedArray,
    ctx: &'a ReadContext,
    session: &'a VortexSession,
}

impl ArrayChildren for SerializedArrayChildren<'_> {
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
        self.ser
            .child(index)
            .decode(dtype, len, self.ctx, self.session)
    }

    fn len(&self) -> usize {
        self.ser.nchildren()
    }
}

impl TryFrom<ByteBuffer> for SerializedArray {
    type Error = VortexError;

    fn try_from(value: ByteBuffer) -> Result<Self, Self::Error> {
        // The final 4 bytes contain the length of the flatbuffer.
        if value.len() < 4 {
            vortex_bail!("SerializedArray buffer is too short");
        }

        // We align each buffer individually, so we remove alignment requirements on the buffer.
        let value = value.aligned(Alignment::none());

        let fb_length = u32::try_from_le_bytes(&value.as_slice()[value.len() - 4..])? as usize;
        if value.len() < 4 + fb_length {
            vortex_bail!("SerializedArray buffer is too short for flatbuffer");
        }

        let fb_offset = value.len() - 4 - fb_length;
        let array_tree = value.slice(fb_offset..fb_offset + fb_length);
        let segment = BufferHandle::new_host(value.slice(0..fb_offset));

        Self::from_flatbuffer_and_segment(array_tree, segment)
    }
}

impl TryFrom<BufferHandle> for SerializedArray {
    type Error = VortexError;

    fn try_from(value: BufferHandle) -> Result<Self, Self::Error> {
        Self::try_from(value.try_to_host_sync()?)
    }
}

// =============================================================================
// Columnar serialization (parallel to SerializedArray, no flatbuffer involved)
// =============================================================================
//
// `SerializedArray` parses a per-chunk flatbuffer (`fba::Array`) and navigates it via
// vtables. That format lives in the data segment's trailing buffer for `FlatLayout` and
// inside the `array_trees` auxiliary segment of `ArrayTreeLayout` files written by older
// builds.
//
// `ColumnarSerializedArray` is the parallel decode entry point for `ArrayTreeLayout` files
// where the consolidated `array_trees` segment uses a columnar struct-of-Lists encoding
// instead of opaque flatbuffer blobs. The plugin contract (`ArrayChildren` trait +
// `plugin.deserialize(dtype, len, metadata, buffers, children, session)`) doesn't care
// which source the metadata/buffers/children come from, so this type implements the same
// decode flow without ever constructing or parsing a flatbuffer.

/// Per-stat raw blob with the inline flatbuffer's precision flag preserved.
///
/// `bytes` is `ScalarValue::to_proto_bytes`-encoded. Pairs with the precision flag because
/// `min` / `max` in `fba::ArrayStats` track exact vs. inexact independently of the value.
#[derive(Debug, Clone)]
pub struct RawStatValue {
    pub bytes: ByteBuffer,
    pub exact: bool,
}

/// Raw, dtype-agnostic snapshot of a node's statistics. Stored in [`ColumnarChunkData`] so
/// the columnar consolidated array_trees segment can carry stats without needing per-node
/// dtypes at materialization time. Conversion to a typed [`StatsSet`] happens at decode
/// time via [`Self::to_stats_set`], when the dtype is known.
///
/// Mirrors the field set of `fba::ArrayStats` exactly so the columnar and inline paths
/// produce equivalent decoded `StatsSet`s.
#[derive(Debug, Clone, Default)]
pub struct RawNodeStats {
    pub min: Option<RawStatValue>,
    pub max: Option<RawStatValue>,
    pub sum: Option<ByteBuffer>,
    pub null_count: Option<u64>,
    pub nan_count: Option<u64>,
    pub uncompressed_size_in_bytes: Option<u64>,
    pub is_constant: Option<bool>,
    pub is_sorted: Option<bool>,
    pub is_strict_sorted: Option<bool>,
}

impl RawNodeStats {
    /// True when no stat slot is populated. Used by the writer to decide whether to record
    /// `None` for the entire node (and emit nulls in every stat column).
    pub fn is_empty(&self) -> bool {
        self.min.is_none()
            && self.max.is_none()
            && self.sum.is_none()
            && self.null_count.is_none()
            && self.nan_count.is_none()
            && self.uncompressed_size_in_bytes.is_none()
            && self.is_constant.is_none()
            && self.is_sorted.is_none()
            && self.is_strict_sorted.is_none()
    }

    /// Snapshot a typed [`StatsSet`] into raw form, mirroring the same selection /
    /// precision handling as the inline flatbuffer writer.
    pub fn from_stats_set(stats: &StatsSet) -> Self {
        use crate::dtype::Nullability;
        use crate::dtype::PType;
        use crate::expr::stats::Precision;
        use crate::expr::stats::Stat;

        let raw_value = |p: Precision<crate::scalar::ScalarValue>| RawStatValue {
            exact: p.is_exact(),
            bytes: ByteBuffer::from(crate::scalar::ScalarValue::to_proto_bytes::<Vec<u8>>(Some(
                &p.into_inner(),
            ))),
        };

        let bool_dtype = DType::Bool(Nullability::NonNullable);
        let u64_dtype: DType = PType::U64.into();

        Self {
            min: stats.get(Stat::Min).map(raw_value),
            max: stats.get(Stat::Max).map(raw_value),
            sum: stats
                .get(Stat::Sum)
                .and_then(Precision::as_exact)
                .map(|sum| {
                    ByteBuffer::from(crate::scalar::ScalarValue::to_proto_bytes::<Vec<u8>>(Some(
                        &sum,
                    )))
                }),
            null_count: stats
                .get_as::<u64>(Stat::NullCount, &u64_dtype)
                .and_then(Precision::as_exact),
            nan_count: stats
                .get_as::<u64>(Stat::NaNCount, &u64_dtype)
                .and_then(Precision::as_exact),
            uncompressed_size_in_bytes: stats
                .get_as::<u64>(Stat::UncompressedSizeInBytes, &u64_dtype)
                .and_then(Precision::as_exact),
            is_constant: stats
                .get_as::<bool>(Stat::IsConstant, &bool_dtype)
                .and_then(Precision::as_exact),
            is_sorted: stats
                .get_as::<bool>(Stat::IsSorted, &bool_dtype)
                .and_then(Precision::as_exact),
            is_strict_sorted: stats
                .get_as::<bool>(Stat::IsStrictSorted, &bool_dtype)
                .and_then(Precision::as_exact),
        }
    }

    /// Hydrate into a typed [`StatsSet`] for a given array dtype, mirroring
    /// [`StatsSet::from_flatbuffer`] (same per-stat dtype lookup, same precision handling).
    pub fn to_stats_set(&self, dtype: &DType, session: &VortexSession) -> VortexResult<StatsSet> {
        use crate::expr::stats::Precision;
        use crate::expr::stats::Stat;
        use crate::scalar::ScalarValue;

        let mut set = StatsSet::default();

        if let Some(raw) = &self.min
            && let Some(stat_dtype) = Stat::Min.dtype(dtype)
            && let Some(value) =
                ScalarValue::from_proto_bytes(raw.bytes.as_slice(), &stat_dtype, session)?
        {
            set.set(
                Stat::Min,
                if raw.exact {
                    Precision::Exact(value)
                } else {
                    Precision::Inexact(value)
                },
            );
        }
        if let Some(raw) = &self.max
            && let Some(stat_dtype) = Stat::Max.dtype(dtype)
            && let Some(value) =
                ScalarValue::from_proto_bytes(raw.bytes.as_slice(), &stat_dtype, session)?
        {
            set.set(
                Stat::Max,
                if raw.exact {
                    Precision::Exact(value)
                } else {
                    Precision::Inexact(value)
                },
            );
        }
        if let Some(raw) = &self.sum
            && let Some(stat_dtype) = Stat::Sum.dtype(dtype)
            && let Some(value) =
                ScalarValue::from_proto_bytes(raw.as_slice(), &stat_dtype, session)?
        {
            set.set(Stat::Sum, Precision::Exact(value));
        }
        if let Some(v) = self.null_count {
            set.set(Stat::NullCount, Precision::Exact(ScalarValue::from(v)));
        }
        if let Some(v) = self.nan_count {
            set.set(Stat::NaNCount, Precision::Exact(ScalarValue::from(v)));
        }
        if let Some(v) = self.uncompressed_size_in_bytes {
            set.set(
                Stat::UncompressedSizeInBytes,
                Precision::Exact(ScalarValue::from(v)),
            );
        }
        if let Some(v) = self.is_constant {
            set.set(Stat::IsConstant, Precision::Exact(ScalarValue::from(v)));
        }
        if let Some(v) = self.is_sorted {
            set.set(Stat::IsSorted, Precision::Exact(ScalarValue::from(v)));
        }
        if let Some(v) = self.is_strict_sorted {
            set.set(Stat::IsStrictSorted, Precision::Exact(ScalarValue::from(v)));
        }
        Ok(set)
    }
}

/// Per-node statistics from the consolidated columnar consolidated form. `None` means the
/// writer didn't persist any stats for that node.
pub type ColumnarNodeStats = Option<RawNodeStats>;

/// Per-chunk slice of the columnar consolidated tree, shared by all `ColumnarSerializedArray`
/// nodes within the chunk via `Arc`.
///
/// Every `Vec` here is "per node" in pre-order traversal of the encoding tree, except
/// `buffer_padding` / `buffer_alignment` / `buffer_length`, which are flat across all nodes
/// (each node owns a contiguous run of `buffers_per_node[i]` entries). The `subtree_sizes`
/// and `buffer_offsets` fields are precomputed at materialization time for O(1) navigation.
#[derive(Debug)]
pub struct ColumnarChunkData {
    /// Encoding id (as an interned u16 in the file's `ArrayContext`) per node.
    pub encoding_ids: Vec<u16>,
    /// Number of direct children of each node.
    pub child_counts: Vec<u8>,
    /// Opaque encoding-specific metadata bytes per node.
    pub node_metadata: Vec<ByteBuffer>,
    /// Number of buffers owned by each node (their descriptors live in the flat arrays
    /// below, starting at `buffer_offsets[i]`).
    pub buffers_per_node: Vec<u16>,
    /// Per-buffer descriptors, concatenated across all nodes in the same pre-order.
    pub buffer_padding: Vec<u16>,
    pub buffer_alignment_exponent: Vec<u8>,
    pub buffer_length: Vec<u32>,
    /// Per-node statistics.
    pub stats: Vec<ColumnarNodeStats>,
    /// Cumulative subtree size starting at each node (subtree_sizes[i] == 1 + sum of
    /// subtree sizes of direct children of i). Precomputed for O(1) `child(idx)`.
    pub subtree_sizes: Vec<u32>,
    /// Cumulative buffer count up to each node (buffer_offsets[i] == sum of
    /// buffers_per_node[0..i]). Precomputed for O(1) buffer slicing.
    pub buffer_offsets: Vec<u32>,
}

impl ColumnarChunkData {
    /// Compute `subtree_sizes` from `child_counts` via a single right-to-left pass.
    ///
    /// This works because in pre-order traversal, a node's subtree occupies a contiguous
    /// range of indices starting at the node, and a node's subtree size is determined by
    /// itself + the sum of its children's subtree sizes.
    fn compute_subtree_sizes(child_counts: &[u8]) -> Vec<u32> {
        let n = child_counts.len();
        let mut sizes = vec![0u32; n];
        // Right-to-left: when we visit node i, all its descendants have already been
        // visited. Walk children by stepping forward by the previously-computed subtree
        // size of each child.
        for i in (0..n).rev() {
            let mut total = 1u32;
            let mut cursor = i + 1;
            for _ in 0..child_counts[i] {
                let child_size = sizes[cursor];
                total += child_size;
                cursor += child_size as usize;
            }
            sizes[i] = total;
        }
        sizes
    }

    /// Compute `buffer_offsets` as a prefix sum of `buffers_per_node`.
    fn compute_buffer_offsets(buffers_per_node: &[u16]) -> Vec<u32> {
        let mut offsets = Vec::with_capacity(buffers_per_node.len());
        let mut acc = 0u32;
        for &n in buffers_per_node {
            offsets.push(acc);
            acc += n as u32;
        }
        offsets
    }

    /// Construct a chunk with auto-computed `subtree_sizes` and `buffer_offsets`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        encoding_ids: Vec<u16>,
        child_counts: Vec<u8>,
        node_metadata: Vec<ByteBuffer>,
        buffers_per_node: Vec<u16>,
        buffer_padding: Vec<u16>,
        buffer_alignment_exponent: Vec<u8>,
        buffer_length: Vec<u32>,
        stats: Vec<ColumnarNodeStats>,
    ) -> VortexResult<Self> {
        let n = encoding_ids.len();
        if child_counts.len() != n
            || node_metadata.len() != n
            || buffers_per_node.len() != n
            || stats.len() != n
        {
            vortex_bail!(
                "ColumnarChunkData per-node columns must all have length {} (got encoding={}, child_counts={}, node_metadata={}, buffers_per_node={}, stats={})",
                n,
                encoding_ids.len(),
                child_counts.len(),
                node_metadata.len(),
                buffers_per_node.len(),
                stats.len(),
            );
        }
        let total_buffers: usize = buffers_per_node.iter().map(|&b| b as usize).sum();
        if buffer_padding.len() != total_buffers
            || buffer_alignment_exponent.len() != total_buffers
            || buffer_length.len() != total_buffers
        {
            vortex_bail!(
                "ColumnarChunkData per-buffer columns must all have length {} (got padding={}, alignment={}, length={})",
                total_buffers,
                buffer_padding.len(),
                buffer_alignment_exponent.len(),
                buffer_length.len(),
            );
        }
        let subtree_sizes = Self::compute_subtree_sizes(&child_counts);
        let buffer_offsets = Self::compute_buffer_offsets(&buffers_per_node);
        Ok(Self {
            encoding_ids,
            child_counts,
            node_metadata,
            buffers_per_node,
            buffer_padding,
            buffer_alignment_exponent,
            buffer_length,
            stats,
            subtree_sizes,
            buffer_offsets,
        })
    }

    /// Number of nodes in the tree.
    pub fn nnodes(&self) -> usize {
        self.encoding_ids.len()
    }
}

/// Parallel to [`SerializedArray`] but sourced from a columnar representation of the
/// encoding tree rather than a flatbuffer.
///
/// Holds a per-chunk `Arc<ColumnarChunkData>` plus a `node_index` that identifies the
/// current node within the tree. `child(idx)` returns a new `ColumnarSerializedArray`
/// pointing at the requested child by computing the child's pre-order index from
/// `subtree_sizes`.
///
/// `decode()` performs the same plugin dispatch as `SerializedArray::decode`, just sourcing
/// metadata/buffers/stats from the columnar chunk data.
#[derive(Clone)]
pub struct ColumnarSerializedArray {
    chunk: Arc<ColumnarChunkData>,
    node_index: usize,
    buffers: Arc<[BufferHandle]>,
}

impl Debug for ColumnarSerializedArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnarSerializedArray")
            .field("encoding_id", &self.encoding_id())
            .field("node_index", &self.node_index)
            .field("nchildren", &self.nchildren())
            .field("nbuffers", &self.nbuffers())
            .finish()
    }
}

impl ColumnarSerializedArray {
    /// Construct a new root-level `ColumnarSerializedArray` for a chunk.
    pub fn new(chunk: Arc<ColumnarChunkData>, buffers: Arc<[BufferHandle]>) -> VortexResult<Self> {
        if chunk.nnodes() == 0 {
            vortex_bail!("ColumnarChunkData must have at least one node");
        }
        Ok(Self {
            chunk,
            node_index: 0,
            buffers,
        })
    }

    /// Slice the data-buffer prefix of a segment into per-buffer handles using the
    /// descriptors in `chunk`, then construct a root-level `ColumnarSerializedArray`.
    ///
    /// Works for segments produced by both [`SegmentMode::Inline`] (which appends a
    /// flatbuffer + length suffix after the data buffers) and [`SegmentMode::DataOnly`]:
    /// the chunk's descriptors only describe the data prefix, so the trailing inline
    /// flatbuffer (if any) is simply ignored.
    pub fn from_segment_and_chunk(
        segment: BufferHandle,
        chunk: Arc<ColumnarChunkData>,
    ) -> VortexResult<Self> {
        let segment = segment.ensure_aligned(Alignment::none())?;
        let n_buffers = chunk.buffer_length.len();
        let mut handles: Vec<BufferHandle> = Vec::with_capacity(n_buffers);
        let mut offset = 0;
        for i in 0..n_buffers {
            offset += chunk.buffer_padding[i] as usize;
            let buffer_len = chunk.buffer_length[i] as usize;
            let alignment = Alignment::from_exponent(chunk.buffer_alignment_exponent[i]);
            let buffer = segment.slice(offset..(offset + buffer_len));
            handles.push(buffer.ensure_aligned(alignment)?);
            offset += buffer_len;
        }
        Self::new(chunk, Arc::from(handles))
    }

    /// Returns the encoding id (as the interned `u16` in the file's `ArrayContext`) of the
    /// current node.
    pub fn encoding_id(&self) -> u16 {
        self.chunk.encoding_ids[self.node_index]
    }

    /// Returns the metadata bytes for the current node.
    pub fn metadata(&self) -> &[u8] {
        self.chunk.node_metadata[self.node_index].as_slice()
    }

    /// Returns the number of direct children of the current node.
    pub fn nchildren(&self) -> usize {
        self.chunk.child_counts[self.node_index] as usize
    }

    /// Returns a `ColumnarSerializedArray` pointing at the `idx`th direct child of the
    /// current node.
    pub fn child(&self, idx: usize) -> ColumnarSerializedArray {
        let n_children = self.nchildren();
        if idx >= n_children {
            vortex_panic!(
                "Invalid child index {} for node with {} children",
                idx,
                n_children
            );
        }
        // Children are laid out in pre-order immediately after the current node. The first
        // child is at node_index + 1; each subsequent child sits at the previous child's
        // index + that child's subtree size.
        let mut cursor = self.node_index + 1;
        for _ in 0..idx {
            cursor += self.chunk.subtree_sizes[cursor] as usize;
        }
        Self {
            chunk: Arc::clone(&self.chunk),
            node_index: cursor,
            buffers: Arc::clone(&self.buffers),
        }
    }

    /// Number of buffers owned by the current node.
    pub fn nbuffers(&self) -> usize {
        self.chunk.buffers_per_node[self.node_index] as usize
    }

    /// Return the slice of buffer handles owned by the current node.
    fn node_buffers(&self) -> VortexResult<&[BufferHandle]> {
        let start = self.chunk.buffer_offsets[self.node_index] as usize;
        let count = self.nbuffers();
        self.buffers.get(start..start + count).ok_or_else(|| {
            vortex_err!(
                "buffer indices {}..{} out of range for {} buffers",
                start,
                start + count,
                self.buffers.len(),
            )
        })
    }

    /// Decode this node into an `ArrayRef` using the same plugin contract as
    /// [`SerializedArray::decode`].
    pub fn decode(
        &self,
        dtype: &DType,
        len: usize,
        ctx: &ReadContext,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let encoding_idx = self.encoding_id();
        let encoding_id = ctx
            .resolve(encoding_idx)
            .ok_or_else(|| vortex_err!("Unknown encoding index: {}", encoding_idx))?;
        let plugin = session
            .arrays()
            .registry()
            .find(&encoding_id)
            .ok_or_else(|| vortex_err!("Unknown encoding: {}", encoding_id))?;

        let buffers = self.node_buffers()?;
        let children = ColumnarSerializedArrayChildren {
            ser: self,
            ctx,
            session,
        };

        let decoded =
            plugin.deserialize(dtype, len, self.metadata(), buffers, &children, session)?;

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
        assert!(
            plugin.is_supported_encoding(&decoded.encoding_id()),
            "Array decoded from {} has incorrect encoding {}",
            encoding_id,
            decoded.encoding_id(),
        );

        // Populate statistics from the columnar chunk data. Hydrate the raw blob now that
        // we know the array's dtype.
        if let Some(raw) = &self.chunk.stats[self.node_index] {
            let stats_set = raw.to_stats_set(dtype, session)?;
            decoded.statistics().set_iter(stats_set.into_iter());
        }

        Ok(decoded)
    }
}

/// Determines the on-disk shape of an array tree leaf's data segment when used alongside
/// the columnar consolidated array_trees segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentMode {
    /// Compat mode. Data buffers + combined flatbuffer (tree + per-node stats) + u32
    /// length suffix. Layout-compatible with `FlatLayout`; any reader that parses a flat
    /// segment will work standalone, with the columnar consolidated acting as a
    /// performance optimization.
    Inline,
    /// Skip-inline mode. Data buffers only, no trailing flatbuffer, no length suffix.
    /// Segments are not self-contained — the consolidated columnar array_trees segment
    /// is the only source of encoding metadata. Smaller files but only readable through
    /// the array-tree-aware reader path.
    DataOnly,
}

/// Walk `array` once and produce both:
/// 1. The data segment buffer list (shape depends on `mode`).
/// 2. A `ColumnarChunkData` capturing the encoding tree + per-node stats + buffer
///    descriptors in the columnar form consumed by [`ColumnarSerializedArray`].
///
/// This is the writer-side entry point used by `ArrayTreeFlatStrategy`. The single walk
/// avoids the previous "serialize for inline AND serialize_array_tree for side channel"
/// double traversal.
pub fn serialize_with_columnar_chunk(
    array: &ArrayRef,
    ctx: &ArrayContext,
    session: &VortexSession,
    options: &SerializeOptions,
    mode: SegmentMode,
) -> VortexResult<(Vec<ByteBuffer>, ColumnarChunkData)> {
    // Single DFS walk: collect per-node columnar data and all buffers in pre-order.
    let mut encoding_ids = Vec::new();
    let mut child_counts = Vec::new();
    let mut node_metadata = Vec::new();
    let mut buffers_per_node = Vec::new();
    let mut stats = Vec::new();
    let mut array_buffers = Vec::new();

    for node in array.depth_first_traversal() {
        let encoding_idx = ctx.intern(&node.encoding_id()).ok_or_else(|| {
            vortex_err!("Array encoding {} not permitted by ctx", node.encoding_id())
        })?;
        encoding_ids.push(encoding_idx);

        let n_children = u8::try_from(node.nchildren())
            .map_err(|_| vortex_err!("Array node has more than u8::MAX children"))?;
        child_counts.push(n_children);

        let metadata_bytes = session.array_serialize(&node)?.ok_or_else(|| {
            vortex_err!(
                "Array {} does not support serialization",
                node.encoding_id()
            )
        })?;
        node_metadata.push(ByteBuffer::from(metadata_bytes));

        let node_bufs = node.buffers();
        let n_buffers = u16::try_from(node_bufs.len())
            .map_err(|_| vortex_err!("Array node has more than u16::MAX buffers"))?;
        buffers_per_node.push(n_buffers);

        // Capture per-node stats. Snapshot the current StatsSet contents and lower them to
        // raw form so the consolidated array_trees segment can carry them without per-node
        // dtypes; the decode path rehydrates with the correct dtype. `None` is recorded
        // when the node has no stats so the read side can distinguish "no stats persisted"
        // from "all stats happen to be empty".
        let stats_set = node.statistics().to_owned();
        let raw = if stats_set.is_empty() {
            None
        } else {
            let raw = RawNodeStats::from_stats_set(&stats_set);
            if raw.is_empty() { None } else { Some(raw) }
        };
        stats.push(raw);

        array_buffers.extend(node_bufs);
    }

    // Per-buffer descriptors, computed with the same padding rules as `serialize()` so
    // the columnar form points at the same byte offsets the inline path uses.
    let fb_buffers = collect_buffer_descriptors(&array_buffers, options)?;
    let buffer_padding: Vec<u16> = fb_buffers.iter().map(|b| b.padding()).collect();
    let buffer_alignment_exponent: Vec<u8> =
        fb_buffers.iter().map(|b| b.alignment_exponent()).collect();
    let buffer_length: Vec<u32> = fb_buffers.iter().map(|b| b.length()).collect();

    let chunk = ColumnarChunkData::new(
        encoding_ids,
        child_counts,
        node_metadata,
        buffers_per_node,
        buffer_padding,
        buffer_alignment_exponent,
        buffer_length,
        stats,
    )?;

    // Assemble the segment buffer list. The data-buffer prefix is identical between
    // Inline and DataOnly modes; the modes differ only in whether the trailing combined
    // flatbuffer + length suffix is appended.
    let max_alignment = array_buffers
        .iter()
        .map(|buf| buf.alignment())
        .chain(iter::once(FlatBuffer::alignment()))
        .max()
        .unwrap_or_else(FlatBuffer::alignment);
    let zeros = ByteBuffer::zeroed(*max_alignment);

    let mut buffers = vec![ByteBuffer::zeroed_aligned(0, max_alignment)];
    let mut pos = options.offset;
    for buffer in &array_buffers {
        if options.include_padding {
            let padding = pos.next_multiple_of(*buffer.alignment()) - pos;
            if padding > 0 {
                pos += padding;
                buffers.push(zeros.slice(0..padding));
            }
        }
        pos += buffer.len();
        buffers.push(buffer.clone().aligned(Alignment::none()));
    }

    if matches!(mode, SegmentMode::Inline) {
        // Inline path: append the combined flatbuffer (tree + stats) and a u32 length
        // suffix so the segment is parseable by the standard `SerializedArray::try_from`
        // path. Bytes here are byte-identical to today's `ArrayRef::serialize` output.
        let fb_buffer = build_array_flatbuffer(ctx, session, array, fb_buffers, false)?;
        let fb_length = fb_buffer.len();
        if options.include_padding {
            let padding = pos.next_multiple_of(*FlatBuffer::alignment()) - pos;
            if padding > 0 {
                buffers.push(zeros.slice(0..padding));
            }
        }
        buffers.push(fb_buffer);
        buffers.push(ByteBuffer::from(
            u32::try_from(fb_length)
                .map_err(|_| {
                    vortex_err!(
                        "Array metadata flatbuffer must fit into u32 for serialization. Array encoding tree is too large."
                    )
                })?
                .to_le_bytes()
                .to_vec(),
        ));
    }

    Ok((buffers, chunk))
}

struct ColumnarSerializedArrayChildren<'a> {
    ser: &'a ColumnarSerializedArray,
    ctx: &'a ReadContext,
    session: &'a VortexSession,
}

impl ArrayChildren for ColumnarSerializedArrayChildren<'_> {
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
        self.ser
            .child(index)
            .decode(dtype, len, self.ctx, self.session)
    }

    fn len(&self) -> usize {
        self.ser.nchildren()
    }
}

#[cfg(test)]
mod columnar_tests {
    use super::*;

    /// Tree shape:
    ///   0 (root, 2 children)
    ///   ├── 1 (leaf)
    ///   └── 2 (1 child)
    ///       └── 3 (leaf)
    /// Subtree sizes: [4, 1, 2, 1].
    #[test]
    fn subtree_sizes_basic() -> VortexResult<()> {
        let child_counts = vec![2u8, 0, 1, 0];
        let sizes = ColumnarChunkData::compute_subtree_sizes(&child_counts);
        assert_eq!(sizes, vec![4, 1, 2, 1]);
        Ok(())
    }

    /// Single-node tree.
    #[test]
    fn subtree_sizes_leaf() -> VortexResult<()> {
        let sizes = ColumnarChunkData::compute_subtree_sizes(&[0u8]);
        assert_eq!(sizes, vec![1]);
        Ok(())
    }

    /// Deeply nested tree (left-skewed):
    ///   0 -> 1 -> 2 -> 3 (leaf)
    /// Subtree sizes: [4, 3, 2, 1].
    #[test]
    fn subtree_sizes_skewed() -> VortexResult<()> {
        let sizes = ColumnarChunkData::compute_subtree_sizes(&[1u8, 1, 1, 0]);
        assert_eq!(sizes, vec![4, 3, 2, 1]);
        Ok(())
    }

    #[test]
    fn buffer_offsets_basic() {
        let offsets = ColumnarChunkData::compute_buffer_offsets(&[2u16, 0, 3, 1]);
        assert_eq!(offsets, vec![0, 2, 2, 5]);
    }

    /// Round-trip a populated `StatsSet` through `RawNodeStats::from_stats_set` and
    /// `to_stats_set` to confirm the dtype-agnostic raw form preserves the same selection
    /// of stats and their values.
    #[test]
    fn raw_node_stats_roundtrip_i32() -> VortexResult<()> {
        use crate::LEGACY_SESSION;
        use crate::dtype::Nullability;
        use crate::dtype::PType;
        use crate::expr::stats::Precision;
        use crate::expr::stats::Stat;
        use crate::scalar::ScalarValue;

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut set = StatsSet::default();
        set.set(Stat::Min, Precision::Exact(ScalarValue::from(-3i32)));
        set.set(Stat::Max, Precision::Inexact(ScalarValue::from(42i32)));
        set.set(Stat::Sum, Precision::Exact(ScalarValue::from(100i64)));
        set.set(Stat::NullCount, Precision::Exact(ScalarValue::from(7u64)));
        set.set(Stat::IsConstant, Precision::Exact(ScalarValue::from(false)));
        set.set(Stat::IsSorted, Precision::Exact(ScalarValue::from(true)));

        let raw = RawNodeStats::from_stats_set(&set);
        assert!(!raw.is_empty());
        // Precision is preserved on min/max.
        assert!(raw.min.as_ref().unwrap().exact);
        assert!(!raw.max.as_ref().unwrap().exact);

        let back = raw.to_stats_set(&dtype, &LEGACY_SESSION)?;
        assert_eq!(
            back.get_as::<i32>(Stat::Min, &dtype),
            Some(Precision::Exact(-3))
        );
        assert_eq!(
            back.get_as::<i32>(Stat::Max, &dtype),
            Some(Precision::Inexact(42))
        );
        assert_eq!(
            back.get_as::<u64>(Stat::NullCount, &PType::U64.into()),
            Some(Precision::Exact(7))
        );
        assert_eq!(
            back.get_as::<bool>(Stat::IsConstant, &DType::Bool(Nullability::NonNullable)),
            Some(Precision::Exact(false))
        );
        assert_eq!(
            back.get_as::<bool>(Stat::IsSorted, &DType::Bool(Nullability::NonNullable)),
            Some(Precision::Exact(true))
        );
        // IsStrictSorted wasn't set; should remain absent.
        assert!(
            back.get(Stat::IsStrictSorted).is_none(),
            "unset stats stay unset"
        );
        Ok(())
    }

    /// Empty stats stay empty across the round trip.
    #[test]
    fn raw_node_stats_empty() {
        let raw = RawNodeStats::from_stats_set(&StatsSet::default());
        assert!(raw.is_empty());
    }

    /// Child navigation: from root (idx 0) of a tree
    ///   0 [2 children]
    ///   ├── 1 [leaf]
    ///   └── 2 [1 child]
    ///       └── 3 [leaf]
    /// expect child(0) -> node 1, child(1) -> node 2. Then from node 2, child(0) -> node 3.
    #[test]
    fn child_navigation() -> VortexResult<()> {
        let chunk = Arc::new(ColumnarChunkData::new(
            vec![0u16, 1, 2, 3],
            vec![2u8, 0, 1, 0],
            vec![ByteBuffer::empty(); 4],
            vec![0u16; 4],
            vec![],
            vec![],
            vec![],
            vec![None; 4],
        )?);
        let root = ColumnarSerializedArray::new(chunk, Arc::new([]))?;
        assert_eq!(root.encoding_id(), 0);
        assert_eq!(root.nchildren(), 2);
        let c0 = root.child(0);
        assert_eq!(c0.encoding_id(), 1);
        assert_eq!(c0.nchildren(), 0);
        let c1 = root.child(1);
        assert_eq!(c1.encoding_id(), 2);
        assert_eq!(c1.nchildren(), 1);
        let c1c0 = c1.child(0);
        assert_eq!(c1c0.encoding_id(), 3);
        assert_eq!(c1c0.nchildren(), 0);
        Ok(())
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use itertools::Itertools as _;
use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ZstdFrameMetadata;
use crate::ZstdMetadata;

// Zstd doesn't support training dictionaries on very few samples.
const MIN_SAMPLES_FOR_DICTIONARY: usize = 8;
type ViewLen = u32;

// Overall approach here:
// Zstd can be used on the whole array (values_per_frame = 0), resulting in a single Zstd
// frame, or it can be used with a dictionary (values_per_frame < # values), resulting in
// multiple Zstd frames sharing a common dictionary. This latter case is helpful if you
// want somewhat faster access to slices or individual rows, allowing us to only
// decompress the necessary frames.

// Visually, during decompression, we have an interval of frames we're
// decompressing and a tighter interval of the slice we actually care about.
// |=============values (all valid elements)==============|
// |<-skipped_uncompressed->|----decompressed-------------|
//                              |------slice-------|
//                              ^                  ^
// |<-slice_uncompressed_start->|                  |
// |<------------slice_uncompressed_stop---------->|
// We then insert these values to the correct position using a primitive array
// constructor.

/// A [`Zstd`]-encoded Vortex array.
pub type ZstdArray = Array<Zstd>;

impl ArrayHash for ZstdData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        match &self.dictionary {
            Some(dict) => {
                true.hash(state);
                dict.array_hash(state, accuracy);
            }
            None => {
                false.hash(state);
            }
        }
        for frame in &self.frames {
            frame.array_hash(state, accuracy);
        }
        self.unsliced_n_rows.hash(state);
        self.slice_start.hash(state);
        self.slice_stop.hash(state);
    }
}

impl ArrayEq for ZstdData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        if !match (&self.dictionary, &other.dictionary) {
            (Some(d1), Some(d2)) => d1.array_eq(d2, accuracy),
            (None, None) => true,
            _ => false,
        } {
            return false;
        }
        if self.frames.len() != other.frames.len() {
            return false;
        }
        for (a, b) in self.frames.iter().zip(&other.frames) {
            if !a.array_eq(b, accuracy) {
                return false;
            }
        }
        self.unsliced_n_rows == other.unsliced_n_rows
            && self.slice_start == other.slice_start
            && self.slice_stop == other.slice_stop
    }
}

impl VTable for Zstd {
    type TypedArrayData = ZstdData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.zstd");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let validity = child_to_validity(slots[0].as_ref(), dtype.nullability());
        data.validate(dtype, len, &validity)
    }

    fn nbuffers(array: ArrayView<'_, Self>) -> usize {
        array.dictionary.is_some() as usize + array.frames.len()
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        if let Some(dict) = &array.dictionary {
            if idx == 0 {
                return BufferHandle::new_host(dict.clone());
            }
            BufferHandle::new_host(array.frames[idx - 1].clone())
        } else {
            BufferHandle::new_host(array.frames[idx].clone())
        }
    }

    fn buffer_name(array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        if array.dictionary.is_some() {
            if idx == 0 {
                Some("dictionary".to_string())
            } else {
                Some(format!("frame_{}", idx - 1))
            }
        } else {
            Some(format!("frame_{idx}"))
        }
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(array.metadata.clone().encode_to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = ZstdMetadata::decode(metadata)?;
        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("ZstdArray expected 0 or 1 child, got {}", children.len());
        };

        let (dictionary_buffer, compressed_buffers) = if metadata.dictionary_size == 0 {
            // no dictionary
            (
                None,
                buffers
                    .iter()
                    .map(|b| b.clone().try_to_host_sync())
                    .collect::<VortexResult<Vec<_>>>()?,
            )
        } else {
            // with dictionary
            (
                Some(buffers[0].clone().try_to_host_sync()?),
                buffers[1..]
                    .iter()
                    .map(|b| b.clone().try_to_host_sync())
                    .collect::<VortexResult<Vec<_>>>()?,
            )
        };

        let slots = smallvec![validity_to_child(&validity, len)];
        let data = ZstdData::new(dictionary_buffer, compressed_buffers, metadata, len);
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let unsliced_validity = child_to_validity(
            array.as_ref().slots()[0].as_ref(),
            array.dtype().nullability(),
        );
        array
            .data()
            .decompress(array.dtype(), &unsliced_validity, ctx)?
            .execute::<ArrayRef>(ctx)
            .map(ExecutionResult::done)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::rules::RULES.evaluate(array, parent, child_idx)
    }
}

#[derive(Clone, Debug)]
pub struct Zstd;

impl Zstd {
    pub fn try_new(dtype: DType, data: ZstdData, validity: Validity) -> VortexResult<ZstdArray> {
        let len = data.len();
        data.validate(&dtype, len, &validity)?;
        let slots = smallvec![validity_to_child(&validity, data.unsliced_n_rows())];
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(Zstd, dtype, len, data).with_slots(slots))
        })
    }

    /// Compress a [`VarBinViewArray`] using Zstd without a dictionary.
    pub fn from_var_bin_view_without_dict(
        vbv: &VarBinViewArray,
        level: i32,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ZstdArray> {
        let validity = vbv.validity()?;
        Self::try_new(
            vbv.dtype().clone(),
            ZstdData::from_var_bin_view_without_dict(vbv, level, values_per_frame, ctx)?,
            validity,
        )
    }

    /// Compress a [`PrimitiveArray`] using Zstd.
    pub fn from_primitive(
        parray: &PrimitiveArray,
        level: i32,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ZstdArray> {
        let validity = parray.validity()?;
        Self::try_new(
            parray.dtype().clone(),
            ZstdData::from_primitive(parray, level, values_per_frame, ctx)?,
            validity,
        )
    }

    /// Compress a [`VarBinViewArray`] using Zstd.
    pub fn from_var_bin_view(
        vbv: &VarBinViewArray,
        level: i32,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ZstdArray> {
        let validity = vbv.validity()?;
        Self::try_new(
            vbv.dtype().clone(),
            ZstdData::from_var_bin_view(vbv, level, values_per_frame, ctx)?,
            validity,
        )
    }

    pub fn decompress(array: &ZstdArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let unsliced_validity = child_to_validity(
            array.as_ref().slots()[0].as_ref(),
            array.dtype().nullability(),
        );
        array
            .data()
            .decompress(array.dtype(), &unsliced_validity, ctx)
    }
}

/// The validity bitmap indicating which elements are non-null.
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["validity"];

#[derive(Clone, Debug)]
pub struct ZstdData {
    pub(crate) dictionary: Option<ByteBuffer>,
    pub(crate) frames: Vec<ByteBuffer>,
    pub(crate) metadata: ZstdMetadata,
    unsliced_n_rows: usize,
    slice_start: usize,
    slice_stop: usize,
}

impl Display for ZstdData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "nrows: {}, slice: {}..{}",
            self.unsliced_n_rows, self.slice_start, self.slice_stop
        )
    }
}

pub struct ZstdDataParts {
    pub dictionary: Option<ByteBuffer>,
    pub frames: Vec<ByteBuffer>,
    pub metadata: ZstdMetadata,
    pub validity: Validity,
    pub n_rows: usize,
    pub slice_start: usize,
    pub slice_stop: usize,
}

/// The parts of a [`ZstdArray`] returned by [`ZstdArray::into_parts`].
#[derive(Debug)]
struct Frames {
    dictionary: Option<ByteBuffer>,
    frames: Vec<ByteBuffer>,
    frame_metas: Vec<ZstdFrameMetadata>,
}

fn choose_max_dict_size(uncompressed_size: usize) -> usize {
    // following recommendations from
    // https://github.com/facebook/zstd/blob/v1.5.5/lib/zdict.h#L190
    // that is, 1/100 the data size, up to 100kB.
    // It appears that zstd can't train dictionaries with <256 bytes.
    (uncompressed_size / 100).clamp(256, 100 * 1024)
}

fn collect_valid_primitive(
    parray: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let mask = parray
        .as_ref()
        .validity()?
        .execute_mask(parray.as_ref().len(), ctx)?;
    let result = parray.filter(mask)?.execute::<PrimitiveArray>(ctx)?;
    Ok(result)
}

fn collect_valid_vbv(
    vbv: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ByteBuffer, Vec<usize>)> {
    let mask = vbv
        .as_ref()
        .validity()?
        .execute_mask(vbv.as_ref().len(), ctx)?;
    let buffer_and_value_byte_indices = match mask.bit_buffer() {
        AllOr::None => (Buffer::empty(), Vec::new()),
        _ => {
            let mut buffer = BufferMut::with_capacity(
                usize::try_from(vbv.nbytes()).vortex_expect("must fit into buffer")
                    + mask.true_count() * size_of::<ViewLen>(),
            );
            let mut value_byte_indices = Vec::new();
            vbv.with_iterator(|iterator| {
                // by flattening, we should omit nulls
                for value in iterator.flatten() {
                    value_byte_indices.push(buffer.len());
                    // here's where we write the string lengths
                    buffer
                        .extend_trusted(ViewLen::try_from(value.len())?.to_le_bytes().into_iter());
                    buffer.extend_from_slice(value);
                }
                Ok::<_, VortexError>(())
            })?;
            (buffer.freeze(), value_byte_indices)
        }
    };
    Ok(buffer_and_value_byte_indices)
}

/// Reconstruct BinaryView structs from length-prefixed byte data.
///
/// The buffer contains interleaved u32 lengths (little-endian) and string data.
/// When the cumulative data exceeds `max_buffer_len`, the buffer is split (zero-copy) into
/// multiple segments so that BinaryView's u32 offsets can address all data.
///
/// Pass [`MAX_BUFFER_LEN`] for `max_buffer_len` in production; a smaller value can be used in
/// tests to exercise the splitting path without allocating >2 GiB.
pub fn reconstruct_views(
    buffer: &ByteBuffer,
    max_buffer_len: usize,
) -> (Vec<ByteBuffer>, Buffer<BinaryView>) {
    let mut views = BufferMut::<BinaryView>::empty();
    let mut buffers = Vec::new();
    let mut segment_start: usize = 0;
    let mut offset = 0;

    while offset < buffer.len() {
        let str_len = ViewLen::from_le_bytes(
            buffer
                .get(offset..offset + size_of::<ViewLen>())
                .vortex_expect("corrupted zstd length")
                .try_into()
                .ok()
                .vortex_expect("must fit ViewLen size"),
        ) as usize;

        let value_data_offset = offset + size_of::<ViewLen>();
        let local_offset = value_data_offset - segment_start;

        if local_offset + str_len > max_buffer_len && offset > segment_start {
            buffers.push(buffer.slice(segment_start..offset));
            segment_start = offset;
        }

        let local_offset = u32::try_from(value_data_offset - segment_start)
            .vortex_expect("local offset within segment must fit in u32");
        let buf_index = u32::try_from(buffers.len()).vortex_expect("buffer index must fit in u32");
        let value = &buffer[value_data_offset..value_data_offset + str_len];
        views.push(BinaryView::make_view(value, buf_index, local_offset));
        offset = value_data_offset + str_len;
    }

    if segment_start < buffer.len() {
        buffers.push(buffer.slice(segment_start..buffer.len()));
    }

    (buffers, views.freeze())
}

impl ZstdData {
    pub fn new(
        dictionary: Option<ByteBuffer>,
        frames: Vec<ByteBuffer>,
        metadata: ZstdMetadata,
        n_rows: usize,
    ) -> Self {
        Self {
            dictionary,
            frames,
            metadata,
            unsliced_n_rows: n_rows,
            slice_start: 0,
            slice_stop: n_rows,
        }
    }

    pub fn validate(&self, dtype: &DType, len: usize, validity: &Validity) -> VortexResult<()> {
        vortex_ensure!(
            matches!(
                dtype,
                DType::Primitive(..) | DType::Binary(_) | DType::Utf8(_)
            ),
            "Unsupported dtype for Zstd array: {dtype}"
        );
        vortex_ensure!(
            self.slice_start <= self.slice_stop,
            "Invalid slice range {}..{}",
            self.slice_start,
            self.slice_stop
        );
        vortex_ensure!(
            self.slice_stop <= self.unsliced_n_rows,
            "Slice stop {} exceeds unsliced row count {}",
            self.slice_stop,
            self.unsliced_n_rows
        );
        vortex_ensure!(
            self.slice_stop - self.slice_start == len,
            "Slice length {} does not match array length {}",
            self.slice_stop - self.slice_start,
            len
        );
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == self.unsliced_n_rows,
                "Validity length {} does not match unsliced row count {}",
                validity_len,
                self.unsliced_n_rows
            );
        }

        match &self.dictionary {
            Some(dictionary) => vortex_ensure!(
                usize::try_from(self.metadata.dictionary_size)? == dictionary.len(),
                "Dictionary size metadata {} does not match buffer size {}",
                self.metadata.dictionary_size,
                dictionary.len()
            ),
            None => vortex_ensure!(
                self.metadata.dictionary_size == 0,
                "Dictionary metadata present without dictionary buffer"
            ),
        }
        vortex_ensure!(
            self.frames.len() == self.metadata.frames.len(),
            "Frame count {} does not match metadata frame count {}",
            self.frames.len(),
            self.metadata.frames.len()
        );

        Ok(())
    }

    pub(crate) fn with_slice(&self, start: usize, stop: usize) -> Self {
        let new_start = self.slice_start + start;
        let new_stop = self.slice_start + stop;

        assert!(
            new_start <= self.slice_stop,
            "new slice start {new_start} exceeds end {}",
            self.slice_stop
        );

        assert!(
            new_stop <= self.slice_stop,
            "new slice stop {new_stop} exceeds end {}",
            self.slice_stop
        );

        Self {
            slice_start: new_start,
            slice_stop: new_stop,
            ..self.clone()
        }
    }

    fn compress_values(
        value_bytes: &ByteBuffer,
        frame_byte_starts: &[usize],
        level: i32,
        values_per_frame: usize,
        n_values: usize,
        use_dictionary: bool,
    ) -> VortexResult<Frames> {
        let n_frames = frame_byte_starts.len();

        // Would-be sample sizes if we end up applying zstd dictionary
        let mut sample_sizes = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let frame_byte_end = frame_byte_starts
                .get(i + 1)
                .copied()
                .unwrap_or(value_bytes.len());
            sample_sizes.push(frame_byte_end - frame_byte_starts[i]);
        }
        debug_assert_eq!(sample_sizes.iter().sum::<usize>(), value_bytes.len());

        let (dictionary, mut compressor) = if !use_dictionary
            || sample_sizes.len() < MIN_SAMPLES_FOR_DICTIONARY
        {
            // no dictionary
            (None, zstd::bulk::Compressor::new(level)?)
        } else {
            // with dictionary
            let max_dict_size = choose_max_dict_size(value_bytes.len());
            let dict = zstd::dict::from_continuous(value_bytes, &sample_sizes, max_dict_size)
                .map_err(|err| VortexError::from(err).with_context("while training dictionary"))?;

            let compressor = zstd::bulk::Compressor::with_dictionary(level, &dict)?;
            (Some(ByteBuffer::from(dict)), compressor)
        };

        let mut frame_metas = vec![];
        let mut frames = vec![];
        for i in 0..n_frames {
            let frame_byte_end = frame_byte_starts
                .get(i + 1)
                .copied()
                .unwrap_or(value_bytes.len());

            let uncompressed = &value_bytes.slice(frame_byte_starts[i]..frame_byte_end);
            let compressed = compressor
                .compress(uncompressed)
                .map_err(|err| VortexError::from(err).with_context("while compressing"))?;
            frame_metas.push(ZstdFrameMetadata {
                uncompressed_size: uncompressed.len() as u64,
                n_values: values_per_frame.min(n_values - i * values_per_frame) as u64,
            });
            frames.push(ByteBuffer::from(compressed));
        }

        Ok(Frames {
            dictionary,
            frames,
            frame_metas,
        })
    }

    /// Creates a ZstdArray from a primitive array.
    ///
    /// # Arguments
    /// * `parray` - The primitive array to compress
    /// * `level` - Zstd compression level (0 = default, negative = fast, positive = better compression)
    /// * `values_per_frame` - Number of values per frame (0 = single frame)
    pub fn from_primitive(
        parray: &PrimitiveArray,
        level: i32,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        Self::from_primitive_impl(parray, level, values_per_frame, true, ctx)
    }

    /// Creates a ZstdArray from a primitive array without using a dictionary.
    ///
    /// This is useful when the compressed data will be decompressed by systems
    /// that don't support ZSTD dictionaries (e.g., nvCOMP on GPU).
    ///
    /// Note: Without a dictionary, each frame is compressed independently.
    /// Dictionaries are trained from sample data from previously seen frames,
    /// to improve compression ratio.
    ///
    /// # Arguments
    /// * `parray` - The primitive array to compress
    /// * `level` - Zstd compression level (0 = default, negative = fast, positive = better compression)
    /// * `values_per_frame` - Number of values per frame (0 = single frame)
    pub fn from_primitive_without_dict(
        parray: &PrimitiveArray,
        level: i32,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        Self::from_primitive_impl(parray, level, values_per_frame, false, ctx)
    }

    fn from_primitive_impl(
        parray: &PrimitiveArray,
        level: i32,
        values_per_frame: usize,
        use_dictionary: bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let byte_width = parray.ptype().byte_width();

        // We compress only the valid elements.
        let values = collect_valid_primitive(parray, ctx)?;
        let n_values = values.len();
        let values_per_frame = if values_per_frame > 0 {
            values_per_frame
        } else {
            n_values
        };

        let value_bytes = values.buffer_handle().try_to_host_sync()?;
        // Align frames to buffer alignment. This is necessary for overaligned buffers.
        let alignment = *value_bytes.alignment();
        let step_width = (values_per_frame * byte_width).div_ceil(alignment) * alignment;

        let frame_byte_starts = (0..n_values * byte_width)
            .step_by(step_width)
            .collect::<Vec<_>>();
        let Frames {
            dictionary,
            frames,
            frame_metas,
        } = Self::compress_values(
            &value_bytes,
            &frame_byte_starts,
            level,
            values_per_frame,
            n_values,
            use_dictionary,
        )?;

        let metadata = ZstdMetadata {
            dictionary_size: dictionary
                .as_ref()
                .map_or(0, |dict| dict.len())
                .try_into()?,
            frames: frame_metas,
        };

        Ok(ZstdData::new(dictionary, frames, metadata, parray.len()))
    }

    /// Creates a ZstdArray from a VarBinView array.
    ///
    /// # Arguments
    /// * `vbv` - The VarBinView array to compress
    /// * `level` - Zstd compression level (0 = default, negative = fast, positive = better compression)
    /// * `values_per_frame` - Number of values per frame (0 = single frame)
    pub fn from_var_bin_view(
        vbv: &VarBinViewArray,
        level: i32,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        Self::from_var_bin_view_impl(vbv, level, values_per_frame, true, ctx)
    }

    /// Creates a ZstdArray from a VarBinView array without using a dictionary.
    ///
    /// This is useful when the compressed data will be decompressed by systems
    /// that don't support ZSTD dictionaries (e.g., nvCOMP on GPU).
    ///
    /// Note: Without a dictionary, each frame is compressed independently.
    /// Dictionaries are trained from sample data from previously seen frames,
    /// to improve compression ratio.
    ///
    /// # Arguments
    /// * `vbv` - The VarBinView array to compress
    /// * `level` - Zstd compression level (0 = default, negative = fast, positive = better compression)
    /// * `values_per_frame` - Number of values per frame (0 = single frame)
    pub fn from_var_bin_view_without_dict(
        vbv: &VarBinViewArray,
        level: i32,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        Self::from_var_bin_view_impl(vbv, level, values_per_frame, false, ctx)
    }

    fn from_var_bin_view_impl(
        vbv: &VarBinViewArray,
        level: i32,
        values_per_frame: usize,
        use_dictionary: bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        // Approach for strings: we prefix each string with its length as a u32.
        // This is the same as what Parquet does. In some cases it may be better
        // to separate the binary data and lengths as two separate streams, but
        // this approach is simpler and can be best in cases when there is
        // mutual information between strings and their lengths.
        // We compress only the valid elements.
        let (value_bytes, value_byte_indices) = collect_valid_vbv(vbv, ctx)?;
        let n_values = value_byte_indices.len();
        let values_per_frame = if values_per_frame > 0 {
            values_per_frame
        } else {
            n_values
        };

        let frame_byte_starts = (0..n_values)
            .step_by(values_per_frame)
            .map(|i| value_byte_indices[i])
            .collect::<Vec<_>>();
        let Frames {
            dictionary,
            frames,
            frame_metas,
        } = Self::compress_values(
            &value_bytes,
            &frame_byte_starts,
            level,
            values_per_frame,
            n_values,
            use_dictionary,
        )?;

        let metadata = ZstdMetadata {
            dictionary_size: dictionary
                .as_ref()
                .map_or(0, |dict| dict.len())
                .try_into()?,
            frames: frame_metas,
        };
        Ok(ZstdData::new(dictionary, frames, metadata, vbv.len()))
    }

    pub fn from_canonical(
        canonical: &Canonical,
        level: i32,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self>> {
        match canonical {
            Canonical::Primitive(parray) => Ok(Some(ZstdData::from_primitive(
                parray,
                level,
                values_per_frame,
                ctx,
            )?)),
            Canonical::VarBinView(vbv) => Ok(Some(ZstdData::from_var_bin_view(
                vbv,
                level,
                values_per_frame,
                ctx,
            )?)),
            _ => Ok(None),
        }
    }

    pub fn from_array(
        array: ArrayRef,
        level: i32,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let canonical = array.execute::<Canonical>(ctx)?;
        Self::from_canonical(&canonical, level, values_per_frame, ctx)?
            .ok_or_else(|| vortex_err!("Zstd can only encode Primitive and VarBinView arrays"))
    }

    fn byte_width(dtype: &DType) -> usize {
        if dtype.is_primitive() {
            dtype.as_ptype().byte_width()
        } else {
            1
        }
    }

    fn decompress(
        &self,
        dtype: &DType,
        unsliced_validity: &Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // To start, we figure out which frames we need to decompress, and with
        // what row offset into the first such frame.
        let byte_width = Self::byte_width(dtype);
        let slice_n_rows = self.slice_stop - self.slice_start;
        let slice_value_indices = unsliced_validity
            .execute_mask(self.unsliced_n_rows, ctx)?
            .valid_counts_for_indices(&[self.slice_start, self.slice_stop]);

        let slice_value_idx_start = slice_value_indices[0];
        let slice_value_idx_stop = slice_value_indices[1];

        let mut frames_to_decompress = vec![];
        let mut value_idx_start = 0;
        let mut uncompressed_size_to_decompress = 0;
        let mut n_skipped_values = 0;
        for (frame, frame_meta) in self.frames.iter().zip(&self.metadata.frames) {
            if value_idx_start >= slice_value_idx_stop {
                break;
            }

            let frame_uncompressed_size = usize::try_from(frame_meta.uncompressed_size)
                .vortex_expect("Uncompressed size must fit in usize");
            let frame_n_values = if frame_meta.n_values == 0 {
                // possibly older primitive-only metadata that just didn't store this
                frame_uncompressed_size / byte_width
            } else {
                usize::try_from(frame_meta.n_values).vortex_expect("frame size must fit usize")
            };

            let value_idx_stop = value_idx_start + frame_n_values;
            if value_idx_stop > slice_value_idx_start {
                // we need this frame
                frames_to_decompress.push(frame);
                uncompressed_size_to_decompress += frame_uncompressed_size;
            } else {
                n_skipped_values += frame_n_values;
            }
            value_idx_start = value_idx_stop;
        }

        // then we actually decompress those frames
        let mut decompressor = if let Some(dictionary) = &self.dictionary {
            zstd::bulk::Decompressor::with_dictionary(dictionary)?
        } else {
            zstd::bulk::Decompressor::new()?
        };
        let mut decompressed = ByteBufferMut::with_capacity_aligned(
            uncompressed_size_to_decompress,
            Alignment::new(byte_width),
        );
        unsafe {
            // safety: we immediately fill all bytes in the following loop,
            // assuming our metadata's uncompressed size is correct
            decompressed.set_len(uncompressed_size_to_decompress);
        }
        let mut uncompressed_start = 0;
        for frame in frames_to_decompress {
            let uncompressed_written = decompressor
                .decompress_to_buffer(frame.as_slice(), &mut decompressed[uncompressed_start..])?;
            uncompressed_start += uncompressed_written;
        }
        if uncompressed_start != uncompressed_size_to_decompress {
            vortex_panic!(
                "Zstd metadata or frames were corrupt; expected {} bytes but decompressed {}",
                uncompressed_size_to_decompress,
                uncompressed_start
            );
        }

        let decompressed = decompressed.freeze();
        // Last, we slice the exact values requested out of the decompressed data.
        let mut slice_validity = unsliced_validity.slice(self.slice_start..self.slice_stop)?;

        // NOTE: this block handles setting the output type when the validity and DType disagree.
        //
        // ZSTD is a compact block compressor, meaning that null values are not stored inline in
        // the data frames. A ZSTD Array that was initialized must always hold onto its full
        // validity bitmap, even if sliced to only include non-null values.
        //
        // We ensure that the validity of the decompressed array ALWAYS matches the validity
        // implied by the DType.
        if !dtype.is_nullable() && !matches!(slice_validity, Validity::NonNullable) {
            assert!(
                matches!(slice_validity, Validity::AllValid),
                "ZSTD array expects to be non-nullable but there are nulls after decompression"
            );

            slice_validity = Validity::NonNullable;
        } else if dtype.is_nullable() && matches!(slice_validity, Validity::NonNullable) {
            slice_validity = Validity::AllValid;
        }
        //
        // END OF IMPORTANT BLOCK
        //

        match dtype {
            DType::Primitive(..) => {
                let slice_values_buffer = decompressed.slice(
                    (slice_value_idx_start - n_skipped_values) * byte_width
                        ..(slice_value_idx_stop - n_skipped_values) * byte_width,
                );
                let primitive = PrimitiveArray::from_values_byte_buffer(
                    slice_values_buffer,
                    dtype.as_ptype(),
                    slice_validity,
                    slice_n_rows,
                );

                Ok(primitive.into_array())
            }
            DType::Binary(_) | DType::Utf8(_) => {
                match slice_validity.execute_mask(slice_n_rows, ctx)?.indices() {
                    AllOr::All => {
                        let (buffers, all_views) = reconstruct_views(&decompressed, MAX_BUFFER_LEN);
                        let valid_views = all_views.slice(
                            slice_value_idx_start - n_skipped_values
                                ..slice_value_idx_stop - n_skipped_values,
                        );

                        // SAFETY: we properly construct the views inside `reconstruct_views`
                        Ok(unsafe {
                            VarBinViewArray::new_unchecked(
                                valid_views,
                                Arc::from(buffers),
                                dtype.clone(),
                                slice_validity,
                            )
                        }
                        .into_array())
                    }
                    AllOr::None => Ok(ConstantArray::new(
                        Scalar::null(dtype.clone()),
                        slice_n_rows,
                    )
                    .into_array()),
                    AllOr::Some(valid_indices) => {
                        let (buffers, all_views) = reconstruct_views(&decompressed, MAX_BUFFER_LEN);
                        let valid_views = all_views.slice(
                            slice_value_idx_start - n_skipped_values
                                ..slice_value_idx_stop - n_skipped_values,
                        );

                        let mut views = BufferMut::<BinaryView>::zeroed(slice_n_rows);
                        for (view, index) in valid_views.into_iter().zip_eq(valid_indices) {
                            views[*index] = view
                        }

                        // SAFETY: we properly construct the views inside `reconstruct_views`
                        Ok(unsafe {
                            VarBinViewArray::new_unchecked(
                                views.freeze(),
                                Arc::from(buffers),
                                dtype.clone(),
                                slice_validity,
                            )
                        }
                        .into_array())
                    }
                }
            }
            _ => vortex_panic!("Unsupported dtype for Zstd array: {}", dtype),
        }
    }

    /// Returns the length of the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.slice_stop - self.slice_start
    }

    /// Returns whether the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slice_stop == self.slice_start
    }

    pub fn into_parts(self, validity: Validity) -> ZstdDataParts {
        ZstdDataParts {
            dictionary: self.dictionary,
            frames: self.frames,
            metadata: self.metadata,
            validity,
            n_rows: self.unsliced_n_rows,
            slice_start: self.slice_start,
            slice_stop: self.slice_stop,
        }
    }

    pub(crate) fn slice_start(&self) -> usize {
        self.slice_start
    }

    pub(crate) fn slice_stop(&self) -> usize {
        self.slice_stop
    }

    pub(crate) fn unsliced_n_rows(&self) -> usize {
        self.unsliced_n_rows
    }
}

impl ValidityVTable<Zstd> for Zstd {
    fn validity(array: ArrayView<'_, Zstd>) -> VortexResult<Validity> {
        let unsliced_validity =
            child_to_validity(array.slots()[0].as_ref(), array.dtype().nullability());
        unsliced_validity.slice(array.slice_start()..array.slice_stop())
    }
}

impl OperationsVTable<Zstd> for Zstd {
    fn scalar_at(
        array: ArrayView<'_, Zstd>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let unsliced_validity =
            child_to_validity(array.slots()[0].as_ref(), array.dtype().nullability());
        let sliced = array.data().with_slice(index, index + 1);
        sliced
            .decompress(array.dtype(), &unsliced_validity, ctx)?
            .execute_scalar(0, ctx)
    }
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests {
    use vortex_buffer::ByteBuffer;

    use super::reconstruct_views;
    use crate::array::BinaryView;

    /// Build a Zstd-style interleaved buffer: [u32-LE length][string bytes] repeated.
    fn make_interleaved(strings: &[&[u8]]) -> ByteBuffer {
        let mut buf = Vec::new();
        for s in strings {
            let len = s.len() as u32;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(s);
        }
        ByteBuffer::copy_from(buf.as_slice())
    }

    #[test]
    fn test_reconstruct_views_no_split() {
        let strings: &[&[u8]] = &[b"hello", b"world"];
        let buf = make_interleaved(strings);
        let (buffers, views) = reconstruct_views(&buf, 1024);

        assert_eq!(buffers.len(), 1);
        assert_eq!(views.len(), 2);
        // Each entry: [u32 len (4 bytes)][data], so offsets are 4 and 4+5+4=13
        assert_eq!(views[0], BinaryView::make_view(b"hello", 0, 4));
        assert_eq!(views[1], BinaryView::make_view(b"world", 0, 13));
    }

    #[test]
    fn test_reconstruct_views_split_across_segments() {
        // "aaaaaaaaaaaaa" (13 bytes) and "bbbbbbbbbbbbb" (13 bytes).
        // Each entry occupies 4 (length prefix) + 13 (data) = 17 bytes.
        // With max_buffer_len=20, the second entry's data (offset 4+13+4=21) exceeds the limit,
        // so it rolls into a second segment.
        let strings: &[&[u8]] = &[b"aaaaaaaaaaaaa", b"bbbbbbbbbbbbb"];
        let buf = make_interleaved(strings);
        let (buffers, views) = reconstruct_views(&buf, 20);

        assert_eq!(buffers.len(), 2);
        assert_eq!(views.len(), 2);
        assert_eq!(views[0], BinaryView::make_view(b"aaaaaaaaaaaaa", 0, 4));
        // Second entry starts a new segment at byte 17 (the length prefix), so local offset = 4.
        assert_eq!(views[1], BinaryView::make_view(b"bbbbbbbbbbbbb", 1, 4));
    }
}

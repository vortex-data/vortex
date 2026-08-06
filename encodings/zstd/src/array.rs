// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::Arc;

use itertools::Itertools as _;
use num_traits::AsPrimitive;
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
use vortex_array::array_slots;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::OffsetBuilderPType;
use vortex_array::match_each_varbin_builder;
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
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use zstd::zstd_safe::WriteBuf;

use crate::ZstdFrameMetadata;
use crate::ZstdMetadata;
use crate::validate_frame_content_size;

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
        let validity = child_to_validity(slots[ZstdSlots::VALIDITY].as_ref(), dtype.nullability());
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        let mut data = array.data().clone();
        if data.dictionary.is_some() {
            let Some((dictionary, frames)) = buffers.split_first() else {
                vortex_bail!("Expected dictionary buffer");
            };
            data.dictionary = Some(dictionary.clone().try_to_host_sync()?);
            data.frames = frames
                .iter()
                .map(|buffer| buffer.clone().try_to_host_sync())
                .collect::<VortexResult<Vec<_>>>()?;
        } else {
            data.frames = buffers
                .iter()
                .map(|buffer| buffer.clone().try_to_host_sync())
                .collect::<VortexResult<Vec<_>>>()?;
        }
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
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
        ZstdSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let unsliced_validity = child_to_validity(
            array.as_ref().slots()[ZstdSlots::VALIDITY].as_ref(),
            array.dtype().nullability(),
        );
        array
            .data()
            .decompress(array.dtype(), &unsliced_validity, ctx)?
            .execute::<ArrayRef>(ctx)
            .map(ExecutionResult::done)
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if let Some(result) =
            match_each_varbin_builder!(builder, |builder| append_to_varbin(array, builder, ctx))
        {
            return result;
        }
        // The two arms here are every builder a `Utf8`/`Binary` dtype has: all four
        // `VarBinBuilder` widths above, and `VarBinViewBuilder` below. There is deliberately no
        // canonicalize-then-append fallback — it would decompress to a `VarBinView` only for
        // `VarBinView::append_to_builder` to reject the same remainder.
        let Some(builder) = builder.as_any_mut().downcast_mut::<VarBinViewBuilder>() else {
            vortex_bail!("append_to_builder for Zstd requires a variable-binary builder")
        };
        append_to_varbinview(array, builder, ctx)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::rules::RULES.evaluate(array, parent, child_idx)
    }
}

fn unsliced_validity(array: ArrayView<'_, Zstd>) -> Validity {
    child_to_validity(
        array.slots()[ZstdSlots::VALIDITY].as_ref(),
        array.dtype().nullability(),
    )
}

/// Copies the decompressed values straight into `builder`'s byte storage.
///
/// The decompressed frames interleave a length prefix with each value, so the bytes have to be
/// compacted; sizing the offsets, byte storage and validity from the slice's own metadata keeps
/// that down to one offset store plus one `memcpy` per value.
fn append_to_varbin<O: OffsetBuilderPType>(
    array: ArrayView<'_, Zstd>,
    builder: &mut VarBinBuilder<O>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    usize: AsPrimitive<O>,
{
    let slice = array
        .data()
        .decompress_slice(array.dtype(), &unsliced_validity(array), ctx)?;
    let mask = slice.validity.execute_mask(slice.n_rows, ctx)?;
    append_slice_to_varbin(&slice, &mask, builder)
}

/// Copies the values `mask` marks as valid out of `slice` and into `builder`.
fn append_slice_to_varbin<O: OffsetBuilderPType>(
    slice: &DecompressedSlice,
    mask: &Mask,
    builder: &mut VarBinBuilder<O>,
) -> VortexResult<()>
where
    usize: AsPrimitive<O>,
{
    let (values, num_bytes) = slice.value_bytes()?;
    // Each value is length-prefixed, so the frames can only be walked in order — which is the
    // order `append_valid_slices` visits the valid rows in. A walk that runs out early hands back
    // its remainder, which the builder rejects as a byte-count mismatch.
    let mut values = ZstdValues::new(values);
    builder.append_valid_slices(num_bytes, mask, |_| values.next_value())
}

/// Hands the decompressed frames to `builder` as data buffers with views built over them.
///
/// The frames already hold the values contiguously, so the views can reference them in place and
/// the only per-row work is building one view; going through the canonical array instead would
/// rewrite every view a second time to rebase its buffer index.
fn append_to_varbinview(
    array: ArrayView<'_, Zstd>,
    builder: &mut VarBinViewBuilder,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let slice = array
        .data()
        .decompress_slice(array.dtype(), &unsliced_validity(array), ctx)?;
    let mask = slice.validity.execute_mask(slice.n_rows, ctx)?;

    // No values were stored, so there is nothing to reference and the frames can be dropped.
    if mask.all_false() {
        builder.append_nulls(slice.n_rows);
        return Ok(());
    }

    // The decompressed frames cover whole frames and so can extend past the requested values on
    // either side. Reconstructing over just the requested region keeps the pushed buffers fully
    // utilized, which is what `append_views_built_at` requires: it hands them to the finished
    // array as they are, without compacting them.
    let value_bytes = slice.bytes.slice(slice.value_byte_range()?);
    // The values only reveal themselves while walking the length-prefixed frames, so the views
    // are built inside the builder's numbering callback rather than from a lengths slice.
    builder.append_views_built_at(&mask, |next_buffer_index| {
        let (buffers, valid_views) =
            try_reconstruct_views(&value_bytes, next_buffer_index, MAX_BUFFER_LEN)?;
        vortex_ensure!(
            valid_views.len() == mask.true_count(),
            "Corrupt zstd metadata: the decompressed frames hold {} values for the {} valid rows \
             of the slice",
            valid_views.len(),
            mask.true_count()
        );

        let views = match mask.bit_buffer() {
            AllOr::All => valid_views,
            AllOr::None => unreachable!("handled above"),
            AllOr::Some(bits) => {
                // Null rows carry an empty view, so scatter the stored values into their rows.
                // Walking the set bits a word at a time avoids materializing the mask's indices,
                // which the views are the only consumer of.
                let mut views = BufferMut::<BinaryView>::zeroed(slice.n_rows);
                let mut valid_row = 0;
                bits.for_each_set_index(|index| {
                    // In bounds: `valid_views.len() == mask.true_count()` was checked above, and
                    // `index < slice.n_rows` because `bits` is the mask over those rows.
                    views[index] = valid_views[valid_row];
                    valid_row += 1;
                });
                views.freeze()
            }
        };
        Ok((buffers, views))
    })
}

#[derive(Clone, Debug)]
/// Zstd array encoding marker.
pub struct Zstd;

impl Zstd {
    /// Construct a [`ZstdArray`] from validated compressed data and validity.
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

    /// Decompress a [`ZstdArray`] into its canonical Vortex representation.
    pub fn decompress(array: &ZstdArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let unsliced_validity = child_to_validity(
            array.as_ref().slots()[ZstdSlots::VALIDITY].as_ref(),
            array.dtype().nullability(),
        );
        array
            .data()
            .decompress(array.dtype(), &unsliced_validity, ctx)
    }
}

#[array_slots(Zstd)]
pub struct ZstdSlots {
    /// The validity bitmap indicating which elements are non-null.
    #[slot(0)]
    pub validity: Option<ArrayRef>,
}

#[derive(Clone, Debug)]
/// Encoding-specific data for a [`ZstdArray`].
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

/// Movable parts of a [`ZstdData`] value plus its validity.
pub struct ZstdDataParts {
    /// Optional zstd dictionary shared by all frames.
    pub dictionary: Option<ByteBuffer>,
    /// Compressed zstd frames.
    pub frames: Vec<ByteBuffer>,
    /// Serialized frame and dictionary metadata.
    pub metadata: ZstdMetadata,
    /// Unsliced validity for the array.
    pub validity: Validity,
    /// Unsliced row count.
    pub n_rows: usize,
    /// Start of this logical slice in unsliced row coordinates.
    pub slice_start: usize,
    /// End of this logical slice in unsliced row coordinates.
    pub slice_stop: usize,
}

/// Compressed ZStd frames and their metadata
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
            let views = vbv.views();
            let buffers = vbv
                .data_buffers()
                .iter()
                .map(|b| b.as_host())
                .collect::<Vec<_>>();
            // skip nulls, writing only valid values
            for (i, view) in views.iter().enumerate() {
                if !mask.value(i) {
                    continue;
                }
                let value = if view.is_inlined() {
                    view.as_inlined().value()
                } else {
                    let view_ref = view.as_view();
                    &buffers[view_ref.buffer_index as usize][view_ref.as_range()]
                };
                value_byte_indices.push(buffer.len());
                // here's where we write the string lengths
                buffer.extend_trusted(ViewLen::try_from(value.len())?.to_le_bytes().into_iter());
                buffer.extend_from_slice(value);
            }
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
    start_buf_index: u32,
    max_buffer_len: usize,
) -> (Vec<ByteBuffer>, Buffer<BinaryView>) {
    let (buffers, views, _) = walk_views(buffer, start_buf_index, max_buffer_len);
    (buffers, views)
}

/// [`reconstruct_views`], but rejecting a buffer that the length prefixes do not tile exactly.
fn try_reconstruct_views(
    buffer: &ByteBuffer,
    start_buf_index: u32,
    max_buffer_len: usize,
) -> VortexResult<(Vec<ByteBuffer>, Buffer<BinaryView>)> {
    match walk_views(buffer, start_buf_index, max_buffer_len) {
        (buffers, views, None) => Ok((buffers, views)),
        (_, _, Some(error)) => Err(error),
    }
}

/// Walks `buffer` until it is exhausted or a length prefix leaves it, returning what was decoded
/// along with the error that stopped the walk.
fn walk_views(
    buffer: &ByteBuffer,
    start_buf_index: u32,
    max_buffer_len: usize,
) -> (Vec<ByteBuffer>, Buffer<BinaryView>, Option<VortexError>) {
    let mut views = BufferMut::<BinaryView>::empty();
    let mut buffers = Vec::new();
    let mut segment_start: usize = 0;
    let mut offset = 0;
    // Only a new segment changes the buffer index, so it is tracked instead of recomputed per view.
    let mut buf_index = start_buf_index;
    let mut error = None;

    while offset < buffer.len() {
        let str_len = match zstd_value_len(buffer.as_slice(), offset) {
            Ok(str_len) => str_len,
            Err(err) => {
                error = Some(err);
                break;
            }
        };

        let value_data_offset = offset + size_of::<ViewLen>();
        let local_offset = value_data_offset - segment_start;

        if local_offset + str_len > max_buffer_len && offset > segment_start {
            buffers.push(buffer.slice(segment_start..offset));
            segment_start = offset;
            let Some(next_index) = buf_index.checked_add(1) else {
                error = Some(vortex_err!("Zstd values need more than u32::MAX buffers"));
                break;
            };
            buf_index = next_index;
        }

        let Ok(local_offset) = u32::try_from(value_data_offset - segment_start) else {
            error = Some(vortex_err!(
                "Zstd value offset {} does not fit in u32; max_buffer_len {max_buffer_len} is too large",
                value_data_offset - segment_start
            ));
            break;
        };
        let Some(value) = buffer.get(value_data_offset..value_data_offset + str_len) else {
            error = Some(vortex_err!(
                "Corrupt zstd value: {str_len} bytes at offset {value_data_offset} run past the \
                 end of the {} byte frame buffer",
                buffer.len()
            ));
            break;
        };
        views.push(BinaryView::make_view(value, buf_index, local_offset));
        offset = value_data_offset + str_len;
    }

    if segment_start < buffer.len() {
        buffers.push(buffer.slice(segment_start..buffer.len()));
    }

    (buffers, views.freeze(), error)
}

/// Narrows the views decoded from the frames down to the values a slice requests.
fn slice_views(
    views: &Buffer<BinaryView>,
    range: Range<usize>,
) -> VortexResult<Buffer<BinaryView>> {
    vortex_ensure!(
        range.end <= views.len(),
        "Corrupt zstd metadata: values {}..{} are out of bounds of the {} values held by the \
         decompressed frames",
        range.start,
        range.end,
        views.len()
    );
    Ok(views.slice(range))
}

/// A zstd output buffer over uninitialized spare capacity.
///
/// `decompress_to_buffer` writes through a raw pointer and reports how many bytes it produced, so
/// it never reads its destination — but handing it a `&mut [u8]` covering uninitialized memory
/// would be undefined behaviour regardless of what it does with it. [`WriteBuf`] is the interface
/// zstd provides for exactly this case, and it keeps the alternative (zeroing the whole buffer
/// before every decompression) off the hot path.
struct UninitDestination<'a> {
    spare: &'a mut [MaybeUninit<u8>],
    filled: usize,
}

impl<'a> UninitDestination<'a> {
    fn new(spare: &'a mut [MaybeUninit<u8>]) -> Self {
        Self { spare, filled: 0 }
    }
}

// SAFETY: `as_mut_ptr` and `capacity` describe the whole spare region, so zstd only ever writes
// within it, and `filled_until` merely records the count it reports. `as_slice` is bounded by that
// count, so it never exposes a byte zstd did not write.
unsafe impl WriteBuf for UninitDestination<'_> {
    fn as_slice(&self) -> &[u8] {
        // SAFETY: zstd reported writing `filled` bytes from the start of `spare`.
        unsafe { std::slice::from_raw_parts(self.spare.as_ptr().cast::<u8>(), self.filled) }
    }

    fn capacity(&self) -> usize {
        self.spare.len()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.spare.as_mut_ptr().cast::<u8>()
    }

    unsafe fn filled_until(&mut self, n: usize) {
        self.filled = n;
    }
}

struct DecompressedSlice {
    bytes: ByteBuffer,
    validity: Validity,
    byte_width: usize,
    n_rows: usize,
    value_idx_start: usize,
    value_idx_stop: usize,
    n_skipped_values: usize,
    /// Number of stored values held by `bytes`, which covers whole frames and so may extend past
    /// the requested slice on either side.
    n_buffered_values: usize,
}

impl DecompressedSlice {
    /// The range of stored values this slice requests, as an index into `bytes`.
    ///
    /// The frame metadata that drives `n_skipped_values` is untrusted, so a frame claiming to hold
    /// values that precede the ones we decompressed is rejected instead of wrapping.
    fn value_range(&self) -> VortexResult<Range<usize>> {
        let start = self
            .value_idx_start
            .checked_sub(self.n_skipped_values)
            .ok_or_else(|| {
                vortex_err!(
                    "Corrupt zstd metadata: skipped frames hold {} values, past the first \
                     requested value {}",
                    self.n_skipped_values,
                    self.value_idx_start
                )
            })?;
        let end = self
            .value_idx_stop
            .checked_sub(self.n_skipped_values)
            .ok_or_else(|| {
                vortex_err!(
                    "Corrupt zstd metadata: skipped frames hold {} values, past the last \
                     requested value {}",
                    self.n_skipped_values,
                    self.value_idx_stop
                )
            })?;
        vortex_ensure!(
            start <= end,
            "Corrupt zstd metadata: value range {start}..{end} is not ascending"
        );
        Ok(start..end)
    }

    /// The bounds within `bytes` of the length-prefixed region that should hold exactly this
    /// slice's values.
    ///
    /// Walking the length prefixes is a dependent load chain, so both ends are derived from the
    /// slice metadata where possible: an unsliced array skips both walks. The far end is then a
    /// claim rather than a checked fact, so a caller must hold the values it reads to the count
    /// from [`Self::value_range`] — with [`ZstdValues`], whose shortfall shows up in the byte total
    /// from [`Self::value_bytes`], or by counting decoded values as [`try_reconstruct_views`] does.
    /// Otherwise a region ending part way through a value passes its trailing bytes off as values.
    fn value_byte_range(&self) -> VortexResult<Range<usize>> {
        let Range { start, end } = self.value_range()?;
        let buffer = self.bytes.as_slice();
        let from = zstd_value_offset(buffer, 0, start)?;
        let to = if end == self.n_buffered_values {
            buffer.len()
        } else {
            zstd_value_offset(buffer, from, end - start)?
        };
        vortex_ensure!(
            from <= to && to <= buffer.len(),
            "Corrupt zstd metadata: values {from}..{to} are out of bounds of the {} byte frame \
             buffer",
            buffer.len()
        );
        Ok(from..to)
    }

    /// The length-prefixed region of `bytes` holding exactly this slice's values, along with the
    /// total size of those values once the length prefixes are dropped.
    fn value_bytes(&self) -> VortexResult<(&[u8], usize)> {
        let range = self.value_byte_range()?;
        let n_values = self.value_range()?.len();
        let buffer = self.bytes.as_slice();
        let bytes = buffer.get(range.clone()).ok_or_else(|| {
            vortex_err!(
                "Corrupt zstd metadata: values {}..{} are out of bounds of the {} byte frame \
                 buffer",
                range.start,
                range.end,
                buffer.len()
            )
        })?;
        // Every value carries a length prefix, so the region must be at least that large.
        let prefix_bytes = n_values.checked_mul(size_of::<ViewLen>()).ok_or_else(|| {
            vortex_err!("Corrupt zstd metadata: value count {n_values} overflows a byte count")
        })?;
        let num_bytes = bytes.len().checked_sub(prefix_bytes).ok_or_else(|| {
            vortex_err!(
                "Corrupt zstd metadata: {n_values} values do not fit in the {} bytes holding them",
                bytes.len()
            )
        })?;
        Ok((bytes, num_bytes))
    }
}

/// Returns the byte offset `count` length-prefixed values past `offset`.
///
/// Each step is bounds-checked by the next prefix read, so only the offset the walk lands on needs
/// a check of its own.
fn zstd_value_offset(buffer: &[u8], mut offset: usize, count: usize) -> VortexResult<usize> {
    for _ in 0..count {
        offset += size_of::<ViewLen>() + zstd_value_len(buffer, offset)?;
    }
    vortex_ensure!(
        offset <= buffer.len(),
        "Corrupt zstd values: walking {count} values ended at offset {offset}, past the end of \
         the {} byte frame buffer",
        buffer.len()
    );
    Ok(offset)
}

/// Reads the length prefix of the value starting at `offset`.
#[inline]
fn zstd_value_len(buffer: &[u8], offset: usize) -> VortexResult<usize> {
    let prefix = buffer
        .get(offset..)
        .and_then(|rest| rest.first_chunk::<{ size_of::<ViewLen>() }>())
        .ok_or_else(|| {
            vortex_err!(
                "Corrupt zstd values: length prefix at offset {offset} runs past the end of the \
                 {} byte frame buffer",
                buffer.len()
            )
        })?;
    Ok(ViewLen::from_le_bytes(*prefix) as usize)
}

/// A forward walk over the values of a length-prefixed region.
struct ZstdValues<'a> {
    buffer: &'a [u8],
    offset: usize,
}

impl<'a> ZstdValues<'a> {
    fn new(buffer: &'a [u8]) -> Self {
        Self { buffer, offset: 0 }
    }

    /// The next value, or every byte the walk could not decode once a prefix leaves the region.
    ///
    /// The remainder is what makes a shortfall visible in the byte total alone, without a second
    /// walk to validate the region up front. A caller sizes that total as the region minus one
    /// prefix per value, so `k` of `n` values decoded leaves it still expecting the `n - k`
    /// prefixes the walk abandoned; the remainder covers those and more, or is empty only because
    /// the region ended exactly and the decoded bytes already fall short. Neither can add up.
    fn next_value(&mut self) -> &'a [u8] {
        let value_start = self.offset + size_of::<ViewLen>();
        let value = zstd_value_len(self.buffer, self.offset)
            .ok()
            .and_then(|len| self.buffer.get(value_start..value_start.checked_add(len)?));
        match value {
            Some(value) => {
                self.offset = value_start + value.len();
                value
            }
            None => &self.buffer[self.offset..],
        }
    }
}

impl ZstdData {
    /// Construct unsliced zstd data from raw frames and metadata.
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

    /// Validate dtype, slice, validity, frame, and dictionary invariants.
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
        for (index, (frame, metadata)) in self.frames.iter().zip(&self.metadata.frames).enumerate()
        {
            validate_frame_content_size(frame.as_slice(), metadata.uncompressed_size, index)?;
        }

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
            let mut compressed = compressor
                .compress(uncompressed)
                .map_err(|err| VortexError::from(err).with_context("while compressing"))?;
            compressed.shrink_to_fit();
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

    /// Compress a supported canonical array into zstd data.
    ///
    /// Returns `Ok(None)` for canonical variants that this encoding does not support.
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

    /// Canonicalize and compress an array into zstd data.
    ///
    /// # Errors
    ///
    /// Returns an error if the array's canonical form is unsupported or compression fails.
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

    fn decompress_slice(
        &self,
        dtype: &DType,
        unsliced_validity: &Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<DecompressedSlice> {
        // To start, we figure out which frames we need to decompress, and with
        // what row offset into the first such frame.
        let byte_width = Self::byte_width(dtype);
        let slice_n_rows = self.slice_stop - self.slice_start;
        let unsliced_mask = unsliced_validity.execute_mask(self.unsliced_n_rows, ctx)?;
        let slice_value_indices =
            unsliced_mask.valid_counts_for_indices(&[self.slice_start, self.slice_stop]);

        let slice_value_idx_start = slice_value_indices[0];
        let slice_value_idx_stop = slice_value_indices[1];

        let mut frames_to_decompress = vec![];
        let mut value_idx_start = 0;
        let mut uncompressed_size_to_decompress = 0usize;
        let mut n_skipped_values = 0;
        let mut n_buffered_values = 0;
        for (frame, frame_meta) in self.frames.iter().zip(&self.metadata.frames) {
            if value_idx_start >= slice_value_idx_stop {
                break;
            }

            let frame_uncompressed_size =
                usize::try_from(frame_meta.uncompressed_size).map_err(|_| {
                    vortex_err!(
                        "Zstd frame uncompressed size {} does not fit in a usize",
                        frame_meta.uncompressed_size
                    )
                })?;
            let frame_n_values = if frame_meta.n_values != 0 {
                usize::try_from(frame_meta.n_values).map_err(|_| {
                    vortex_err!(
                        "Zstd frame value count {} does not fit in a usize",
                        frame_meta.n_values
                    )
                })?
            } else if dtype.is_primitive() {
                // Possibly older primitive-only metadata that just didn't store this. Fixed-width
                // values make the byte count an exact value count.
                frame_uncompressed_size / byte_width
            } else {
                // The same fallback would read a byte count as a value count for variable-width
                // values, which misattributes values to frames. A single frame holds every stored
                // value, so that case is still recoverable; anything else is not.
                vortex_ensure!(
                    self.frames.len() == 1,
                    "Zstd frame metadata for a variable-width array is missing its value count"
                );
                unsliced_mask.true_count()
            };

            // Bounding the running total also bounds the two accumulators below, which partition
            // it between the frames we keep and the ones we skip.
            let value_idx_stop = value_idx_start.checked_add(frame_n_values).ok_or_else(|| {
                vortex_err!("Corrupt zstd metadata: frame value counts overflow a usize")
            })?;
            if value_idx_stop > slice_value_idx_start {
                // we need this frame
                frames_to_decompress.push(frame);
                uncompressed_size_to_decompress = uncompressed_size_to_decompress
                    .checked_add(frame_uncompressed_size)
                    .ok_or_else(|| {
                        vortex_err!("Corrupt zstd metadata: frame sizes overflow a usize")
                    })?;
                n_buffered_values += frame_n_values;
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
        let mut uncompressed_start = 0;
        for frame in frames_to_decompress {
            // Decompress straight into the spare capacity. Each frame gets only the region after
            // the ones before it, bounded by the size the metadata declared, so a frame that
            // expands further than advertised is refused by zstd rather than overrunning.
            let mut destination = UninitDestination::new(
                &mut decompressed.spare_capacity_mut()
                    [uncompressed_start..uncompressed_size_to_decompress],
            );
            uncompressed_start +=
                decompressor.decompress_to_buffer(frame.as_slice(), &mut destination)?;
        }
        if uncompressed_start != uncompressed_size_to_decompress {
            vortex_bail!(
                "Zstd metadata or frames were corrupt; expected {} bytes but decompressed {}",
                uncompressed_size_to_decompress,
                uncompressed_start
            );
        }
        // SAFETY: the loop above decompressed exactly `uncompressed_start` bytes into the front of
        // the spare capacity, and the check above pins that to the requested length.
        unsafe { decompressed.set_len(uncompressed_start) };

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
            vortex_ensure!(
                matches!(slice_validity, Validity::AllValid),
                "ZSTD array expects to be non-nullable but there are nulls after decompression"
            );

            slice_validity = Validity::NonNullable;
        } else if dtype.is_nullable() && matches!(slice_validity, Validity::NonNullable) {
            slice_validity = Validity::AllValid;
        }
        // END OF IMPORTANT BLOCK
        //

        Ok(DecompressedSlice {
            bytes: decompressed,
            validity: slice_validity,
            byte_width,
            n_rows: slice_n_rows,
            value_idx_start: slice_value_idx_start,
            value_idx_stop: slice_value_idx_stop,
            n_skipped_values,
            n_buffered_values,
        })
    }

    fn decompress(
        &self,
        dtype: &DType,
        unsliced_validity: &Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let slice = self.decompress_slice(dtype, unsliced_validity, ctx)?;
        match dtype {
            DType::Primitive(..) => {
                let Range { start, end } = slice.value_range()?;
                let byte_range = start
                    .checked_mul(slice.byte_width)
                    .zip(end.checked_mul(slice.byte_width))
                    .filter(|(_, byte_stop)| *byte_stop <= slice.bytes.len())
                    .map(|(byte_start, byte_stop)| byte_start..byte_stop)
                    .ok_or_else(|| {
                        vortex_err!(
                            "Corrupt zstd metadata: values {start}..{end} of {} bytes each are \
                             out of bounds of the {} byte frame buffer",
                            slice.byte_width,
                            slice.bytes.len()
                        )
                    })?;
                let slice_values_buffer = slice.bytes.slice(byte_range);
                let primitive = PrimitiveArray::from_values_byte_buffer(
                    slice_values_buffer,
                    dtype.as_ptype(),
                    slice.validity,
                    slice.n_rows,
                    ctx,
                );

                Ok(primitive.into_array())
            }
            DType::Binary(_) | DType::Utf8(_) => {
                match slice.validity.execute_mask(slice.n_rows, ctx)?.indices() {
                    AllOr::All => {
                        let (buffers, all_views) =
                            try_reconstruct_views(&slice.bytes, 0, MAX_BUFFER_LEN)?;
                        let valid_views = slice_views(&all_views, slice.value_range()?)?;

                        // SAFETY: we properly construct the views inside `reconstruct_views`
                        Ok(unsafe {
                            VarBinViewArray::new_unchecked(
                                valid_views,
                                Arc::from(buffers),
                                dtype.clone(),
                                slice.validity,
                            )
                        }
                        .into_array())
                    }
                    AllOr::None => Ok(ConstantArray::new(
                        Scalar::null(dtype.clone()),
                        slice.n_rows,
                    )
                    .into_array()),
                    AllOr::Some(valid_indices) => {
                        let (buffers, all_views) =
                            try_reconstruct_views(&slice.bytes, 0, MAX_BUFFER_LEN)?;
                        let valid_views = slice_views(&all_views, slice.value_range()?)?;

                        let mut views = BufferMut::<BinaryView>::zeroed(slice.n_rows);
                        for (view, index) in valid_views.into_iter().zip_eq(valid_indices) {
                            views[*index] = view
                        }

                        // SAFETY: we properly construct the views inside `reconstruct_views`
                        Ok(unsafe {
                            VarBinViewArray::new_unchecked(
                                views.freeze(),
                                Arc::from(buffers),
                                dtype.clone(),
                                slice.validity,
                            )
                        }
                        .into_array())
                    }
                }
            }
            _ => vortex_bail!("Unsupported dtype for Zstd array: {}", dtype),
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

    /// Split this data into movable parts, attaching the supplied validity.
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
        let unsliced_validity = child_to_validity(
            array.slots()[ZstdSlots::VALIDITY].as_ref(),
            array.dtype().nullability(),
        );
        unsliced_validity.slice(array.slice_start()..array.slice_stop())
    }
}

impl OperationsVTable<Zstd> for Zstd {
    fn scalar_at(
        array: ArrayView<'_, Zstd>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let unsliced_validity = child_to_validity(
            array.slots()[ZstdSlots::VALIDITY].as_ref(),
            array.dtype().nullability(),
        );
        let sliced = array.data().with_slice(index, index + 1);
        sliced
            .decompress(array.dtype(), &unsliced_validity, ctx)?
            .execute_scalar(0, ctx)
    }
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::varbin::VarBinArrayExt as _;
    use vortex_array::builders::VarBinBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::validity::Validity;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::DecompressedSlice;
    use super::ViewLen;
    use super::append_slice_to_varbin;
    use super::reconstruct_views;
    use super::try_reconstruct_views;
    use super::zstd_value_len;
    use super::zstd_value_offset;
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

    /// A slice over `bytes` that requests `value_idx_start..value_idx_stop`.
    fn decompressed_slice(
        bytes: ByteBuffer,
        value_idx_start: usize,
        value_idx_stop: usize,
        n_skipped_values: usize,
        n_buffered_values: usize,
    ) -> DecompressedSlice {
        DecompressedSlice {
            bytes,
            validity: Validity::NonNullable,
            byte_width: 1,
            n_rows: value_idx_stop - value_idx_start,
            value_idx_start,
            value_idx_stop,
            n_skipped_values,
            n_buffered_values,
        }
    }

    #[test]
    fn test_reconstruct_views_no_split() {
        let strings: &[&[u8]] = &[b"hello", b"world"];
        let buf = make_interleaved(strings);
        let (buffers, views) = reconstruct_views(&buf, 0, 1024);

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
        let (buffers, views) = reconstruct_views(&buf, 0, 20);

        assert_eq!(buffers.len(), 2);
        assert_eq!(views.len(), 2);
        assert_eq!(views[0], BinaryView::make_view(b"aaaaaaaaaaaaa", 0, 4));
        // Second entry starts a new segment at byte 17 (the length prefix), so local offset = 4.
        assert_eq!(views[1], BinaryView::make_view(b"bbbbbbbbbbbbb", 1, 4));
    }

    /// A buffer whose last entry claims more bytes than remain, as corrupt frame data would.
    fn make_overrunning() -> ByteBuffer {
        let mut buf = Vec::new();
        buf.extend_from_slice(&5u32.to_le_bytes());
        buf.extend_from_slice(b"hello");
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(b"ab");
        ByteBuffer::copy_from(buf.as_slice())
    }

    #[test]
    fn test_reconstruct_views_rejects_overrunning_value() {
        let buf = make_overrunning();
        assert!(try_reconstruct_views(&buf, 0, 1024).is_err());

        // The lenient walk keeps the decodable prefix instead of panicking.
        let (buffers, views) = reconstruct_views(&buf, 0, 1024);
        assert_eq!(buffers.len(), 1);
        assert_eq!(views.len(), 1);
        assert_eq!(views[0], BinaryView::make_view(b"hello", 0, 4));
    }

    #[test]
    fn test_reconstruct_views_rejects_truncated_length_prefix() {
        // A trailing partial length prefix cannot start a value.
        let buf =
            ByteBuffer::copy_from([5u8, 0, 0, 0, b'h', b'e', b'l', b'l', b'o', 1, 0].as_ref());
        assert!(try_reconstruct_views(&buf, 0, 1024).is_err());
        assert_eq!(reconstruct_views(&buf, 0, 1024).1.len(), 1);
    }

    #[rstest]
    #[case::truncated_buffer(&[0u8, 0, 0], 0)]
    #[case::truncated_tail(&[4u8, 0, 0, 0], 2)]
    #[case::offset_at_end(&[4u8, 0, 0, 0], 4)]
    #[case::offset_past_end(&[4u8, 0, 0, 0], 64)]
    fn test_zstd_value_len_rejects_out_of_bounds(#[case] buffer: &[u8], #[case] offset: usize) {
        assert!(zstd_value_len(buffer, offset).is_err());
    }

    #[test]
    fn test_zstd_value_offset_rejects_walking_past_the_end() -> VortexResult<()> {
        let buf = make_interleaved(&[b"hello", b"world"]);
        assert_eq!(zstd_value_offset(buf.as_slice(), 0, 2)?, buf.len());
        // Only two values are stored, so the third step leaves the buffer.
        assert!(zstd_value_offset(buf.as_slice(), 0, 3).is_err());
        Ok(())
    }

    #[test]
    fn test_value_range_rejects_skipping_past_the_requested_values() {
        // Frame metadata claiming more skipped values than the slice starts at would wrap.
        let slice = decompressed_slice(make_interleaved(&[b"hello"]), 2, 3, 4, 1);
        assert!(slice.value_range().is_err());
        assert!(slice.value_bytes().is_err());
    }

    #[rstest]
    // The buffered value count matches, so both ends come from the metadata.
    #[case::exact_metadata(2)]
    // It does not, so the far end is walked instead.
    #[case::walked_end(9)]
    fn test_value_bytes_totals_the_stored_values(
        #[case] n_buffered_values: usize,
    ) -> VortexResult<()> {
        let buf = make_interleaved(&[b"hello", b"world"]);
        let slice = decompressed_slice(buf.clone(), 0, 2, 0, n_buffered_values);
        let (bytes, num_bytes) = slice.value_bytes()?;
        assert_eq!(bytes, buf.as_slice());
        assert_eq!(num_bytes, buf.len() - 2 * size_of::<ViewLen>());
        Ok(())
    }

    #[test]
    fn test_value_bytes_rejects_more_values_than_the_buffer_holds() {
        // Frame metadata claims nine values but only two are stored.
        let slice = decompressed_slice(make_interleaved(&[b"hello", b"world"]), 0, 5, 0, 9);
        assert!(slice.value_bytes().is_err());
    }

    #[test]
    fn test_append_to_varbin_copies_the_stored_values() -> VortexResult<()> {
        let slice = decompressed_slice(make_interleaved(&[b"hello", b"world"]), 0, 2, 0, 2);
        let mut builder = VarBinBuilder::<i32>::new(DType::Utf8(NonNullable));
        append_slice_to_varbin(&slice, &Mask::new_true(2), &mut builder)?;

        let appended = builder.finish_into_varbin();
        assert_eq!(appended.bytes_at(0).as_slice(), b"hello");
        assert_eq!(appended.bytes_at(1).as_slice(), b"world");
        Ok(())
    }

    #[test]
    fn test_append_to_varbin_rejects_a_dangling_length_prefix() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(&3u32.to_le_bytes());
        buffer.extend_from_slice(b"cat");
        // A prefix with no value after it. It takes up exactly the four bytes the missing value's
        // own prefix would have, so treating that value as empty still totals the byte count the
        // metadata implies and appends ["cat", ""].
        buffer.extend_from_slice(&1u32.to_le_bytes());

        let slice = decompressed_slice(ByteBuffer::copy_from(buffer.as_slice()), 0, 2, 0, 2);
        let mut builder = VarBinBuilder::<i32>::new(DType::Utf8(NonNullable));
        assert!(append_slice_to_varbin(&slice, &Mask::new_true(2), &mut builder).is_err());
        // The builder rejected the values before committing any of them.
        assert_eq!(builder.finish_into_varbin().len(), 0);
    }
}

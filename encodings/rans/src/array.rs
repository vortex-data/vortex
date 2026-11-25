// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`Rans`] array definition, [`VTable`], [`ValidityVTable`], and [`OperationsVTable`].

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::require_validity;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;

use crate::codec::BLOCK_HEADER_BYTES;
use crate::codec::DEFAULT_BLOCK_SHIFT;
use crate::codec::MAX_BLOCK_VALUES;
use crate::codec::NUM_LANES;
use crate::codec::RansDecodeTable;
use crate::codec::decode_block_stripes;
use crate::codec::decode_block_stripes_into;
use crate::codec::encode_primitive_slice;
use crate::codec::validate_block_headers;
use crate::codec::validate_block_shift;
use crate::plugin::rans_id;

/// A [`Rans`]-encoded Vortex array.
pub type RansArray = Array<Rans>;

/// Child slots of a [`RansArray`].
#[array_slots(Rans)]
pub struct RansSlots {
    /// Optional validity bitmap child in memory (always emitted at wire slot `0` when serialized).
    #[slot(0)]
    pub validity_child: Option<ArrayRef>,
}

/// Marker struct for the `vortex.rans` encoding.
#[derive(Clone, Debug)]
pub struct Rans;

/// Encoding-specific state and buffers for a [`RansArray`].
#[derive(Clone, Debug)]
pub struct RansData {
    pub(crate) offset: u16,
    pub(crate) block_shift: u8,
    pub(crate) payload_base_offset: u32,
    pub(crate) has_extra_bits: bool,
    pub(crate) sym_freqs: Arc<[(u8, u16)]>,
    pub(crate) decode_table: Arc<RansDecodeTable>,
    pub(crate) block_headers: BufferHandle,
    pub(crate) payload: BufferHandle,
}

impl Display for RansData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "offset: {}, block_shift: {}, extra_bits: {}, syms: {}",
            self.offset,
            self.block_shift,
            self.has_extra_bits,
            self.sym_freqs.len()
        )
    }
}

impl ArrayHash for RansData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.offset.hash(state);
        self.block_shift.hash(state);
        self.payload_base_offset.hash(state);
        self.has_extra_bits.hash(state);
        self.sym_freqs.hash(state);
        self.block_headers.array_hash(state, accuracy);
        self.payload.array_hash(state, accuracy);
    }
}

impl ArrayEq for RansData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.offset == other.offset
            && self.block_shift == other.block_shift
            && self.payload_base_offset == other.payload_base_offset
            && self.has_extra_bits == other.has_extra_bits
            && self.sym_freqs == other.sym_freqs
            && self.block_headers.array_eq(&other.block_headers, accuracy)
            && self.payload.array_eq(&other.payload, accuracy)
    }
}

/// Number of `2^block_shift`-element blocks spanned by `offset + len` elements (`0` when `len == 0`).
#[inline]
pub(crate) fn num_blocks_for(offset: u16, len: usize, block_shift: u8) -> usize {
    if len == 0 {
        0
    } else {
        (usize::from(offset) + len).div_ceil(1usize << block_shift)
    }
}

impl RansData {
    /// Constructs [`RansData`] from validated components and precomputed decode table.
    #[expect(clippy::too_many_arguments)]
    pub fn try_new(
        offset: u16,
        block_shift: u8,
        payload_base_offset: u32,
        has_extra_bits: bool,
        sym_freqs: Arc<[(u8, u16)]>,
        decode_table: Arc<RansDecodeTable>,
        block_headers: BufferHandle,
        payload: BufferHandle,
    ) -> VortexResult<Self> {
        let block_values = validate_block_shift(block_shift)?;
        vortex_ensure!(
            usize::from(offset) < block_values,
            "Rans offset must be < {block_values} (block_shift={block_shift}), got {offset}"
        );
        Ok(Self {
            offset,
            block_shift,
            payload_base_offset,
            has_extra_bits,
            sym_freqs,
            decode_table,
            block_headers,
            payload,
        })
    }

    pub(crate) fn validate(
        &self,
        dtype: &DType,
        len: usize,
        validity: &Validity,
    ) -> VortexResult<()> {
        vortex_ensure!(
            dtype.is_int(),
            "RansArray requires an integer dtype, got {dtype}"
        );
        vortex_ensure_eq!(dtype.nullability(), validity.nullability());
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure_eq!(
                validity_len,
                len,
                "RansArray validity length must match array length"
            );
        }
        let block_values = validate_block_shift(self.block_shift)?;
        vortex_ensure!(
            usize::from(self.offset) < block_values,
            "RansArray offset {} must be < {block_values} (block_shift={})",
            self.offset,
            self.block_shift
        );

        let expected_blocks = num_blocks_for(self.offset, len, self.block_shift);
        vortex_ensure_eq!(
            self.block_headers.len(),
            expected_blocks * BLOCK_HEADER_BYTES,
            "RansArray block_headers buffer size mismatch"
        );

        if self.block_headers.is_on_host() && self.payload.is_on_host() {
            validate_block_headers(
                self.block_headers.as_host().as_ref(),
                self.payload.len(),
                self.payload_base_offset,
                expected_blocks,
                self.has_extra_bits,
            )?;
        }
        Ok(())
    }

    /// Element offset (`0..2^block_shift`) within the first block.
    #[inline]
    pub fn offset(&self) -> u16 {
        self.offset
    }

    /// Log2 of the number of elements per independent rANS block (`10`, `11`, or `12`).
    #[inline]
    pub fn block_shift(&self) -> u8 {
        self.block_shift
    }

    /// Number of elements per independent rANS block (`1,024`, `2,048`, or `4,096`).
    #[inline]
    pub fn block_values(&self) -> usize {
        1usize << self.block_shift
    }

    /// Base byte offset subtracted from `block_headers` offsets into `payload`.
    #[inline]
    pub fn payload_base_offset(&self) -> u32 {
        self.payload_base_offset
    }

    /// Whether any active symbol in the chunk has `ebits > 0`.
    #[inline]
    pub fn has_extra_bits(&self) -> bool {
        self.has_extra_bits
    }

    /// Active `(sym, freq)` entries summing to `2048`.
    #[inline]
    pub fn sym_freqs(&self) -> &[(u8, u16)] {
        &self.sym_freqs
    }

    /// Shared `2,048`-slot decode lookup table.
    #[inline]
    pub fn decode_table(&self) -> &Arc<RansDecodeTable> {
        &self.decode_table
    }

    /// Per-block 8-byte headers buffer.
    #[inline]
    pub fn block_headers(&self) -> &BufferHandle {
        &self.block_headers
    }

    /// Compressed rANS states, `u16` refill words, and `u64` raw-bits payload buffer.
    #[inline]
    pub fn payload(&self) -> &BufferHandle {
        &self.payload
    }
}

/// Typed extension accessors for [`RansArray`] and `ArrayView<'_, Rans>`.
pub trait RansArrayExt: TypedArrayRef<Rans> {
    /// Element offset (`0..2^block_shift`) within the first block.
    #[inline]
    fn offset(&self) -> u16 {
        RansData::offset(self)
    }

    /// Log2 of the number of elements per independent rANS block (`10`, `11`, or `12`).
    #[inline]
    fn block_shift(&self) -> u8 {
        RansData::block_shift(self)
    }

    /// Number of elements per independent rANS block (`1,024`, `2,048`, or `4,096`).
    #[inline]
    fn block_values(&self) -> usize {
        1usize << self.block_shift()
    }

    /// Base byte offset subtracted from `block_headers` offsets into `payload`.
    #[inline]
    fn payload_base_offset(&self) -> u32 {
        RansData::payload_base_offset(self)
    }

    /// Whether any active symbol has `ebits > 0`.
    #[inline]
    fn has_extra_bits(&self) -> bool {
        RansData::has_extra_bits(self)
    }

    /// Active `(sym, freq)` entries summing to `2048`.
    #[inline]
    fn sym_freqs(&self) -> &[(u8, u16)] {
        RansData::sym_freqs(self)
    }

    /// Shared `2,048`-slot decode lookup table.
    #[inline]
    fn decode_table(&self) -> &Arc<RansDecodeTable> {
        RansData::decode_table(self)
    }

    /// Per-block 8-byte headers buffer.
    #[inline]
    fn block_headers(&self) -> &BufferHandle {
        RansData::block_headers(self)
    }

    /// Compressed payload buffer.
    #[inline]
    fn payload(&self) -> &BufferHandle {
        RansData::payload(self)
    }

    /// Reconstructed [`Validity`] from `validity_child`.
    #[inline]
    fn rans_validity(&self) -> Validity {
        child_to_validity(self.validity_child(), self.dtype().nullability())
    }
}

impl<T: TypedArrayRef<Rans>> RansArrayExt for T {}

impl Rans {
    /// Constructs a validated [`RansArray`] from its parts.
    pub fn try_new(
        dtype: DType,
        len: usize,
        data: RansData,
        validity: Validity,
    ) -> VortexResult<RansArray> {
        let slots = RansSlots {
            validity_child: validity_to_child(&validity, len),
        }
        .into_slots();
        Array::try_from_parts(ArrayParts::new(Rans, dtype, len, data, slots))
    }

    /// Encodes a non-negative integer [`PrimitiveArray`] into a 32-lane `Pow2-Bin + Raw-Bits`
    /// [`RansArray`] using [`DEFAULT_BLOCK_SHIFT`] (`4,096` values per block).
    ///
    /// Negative signed inputs return an error (wrap with `ZigZag` first).
    pub fn encode(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<RansArray> {
        Self::encode_with_block_shift(array, DEFAULT_BLOCK_SHIFT, ctx)
    }

    /// Encodes a non-negative integer [`PrimitiveArray`] into a 32-lane `Pow2-Bin + Raw-Bits`
    /// [`RansArray`] with `2^block_shift` values per independent block (`block_shift in 10..=12`).
    ///
    /// Negative signed inputs return an error (wrap with `ZigZag` first).
    pub fn encode_with_block_shift(
        array: &PrimitiveArray,
        block_shift: u8,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RansArray> {
        validate_block_shift(block_shift)?;
        let dtype = array.dtype().clone();
        vortex_ensure!(
            dtype.is_int(),
            "Rans::encode requires an integer PrimitiveArray, got {dtype}"
        );
        let ptype = array.ptype();
        let len = array.len();
        let validity = array.validity()?;
        let mask = validity.execute_mask(len, ctx)?;

        let encoded = match_each_integer_ptype!(ptype, |T| {
            encode_primitive_slice::<T>(
                array.as_slice::<T>(),
                ptype.is_signed_int(),
                &mask,
                block_shift,
            )?
        });

        let data = RansData::try_new(
            0,
            block_shift,
            0,
            encoded.has_extra_bits,
            Arc::from(encoded.sym_freqs),
            Arc::new(encoded.decode_table),
            BufferHandle::new_host(encoded.block_headers),
            BufferHandle::new_host(encoded.payload),
        )?;
        let out = Self::try_new(dtype, len, data, validity)?;
        out.statistics().inherit_from(array.statistics());
        Ok(out)
    }
}

impl VTable for Rans {
    type TypedArrayData = RansData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        rans_id()
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), RansSlots::COUNT);
        let validity = child_to_validity(
            RansSlotsView::from_slots(slots).validity_child,
            dtype.nullability(),
        );
        data.validate(dtype, len, &validity)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        2
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.block_headers().clone(),
            1 => array.payload().clone(),
            _ => vortex_panic!("RansArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("block_headers".to_string()),
            1 => Some("payload".to_string()),
            _ => None,
        }
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure_eq!(buffers.len(), 2);
        let mut data = array.data().clone();
        data.block_headers = buffers[0].clone();
        data.payload = buffers[1].clone();
        Ok(ArrayParts::new(
            self.clone(),
            array.dtype().clone(),
            array.len(),
            data,
            array.slots().iter().cloned().collect(),
        ))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        RansSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        require_validity!(array, RansSlots::VALIDITY_CHILD);
        let decoded = decompress_rans(array.as_view(), ctx)?;
        Ok(ExecutionResult::done(decoded.into_array()))
    }
}

impl ValidityVTable<Rans> for Rans {
    fn validity(array: ArrayView<'_, Rans>) -> VortexResult<Validity> {
        Ok(array.rans_validity())
    }
}

impl OperationsVTable<Rans> for Rans {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Rans>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let ptype = array.dtype().as_ptype();
        let nullability = array.dtype().nullability();
        Ok(match_each_integer_ptype!(ptype, |T| {
            scalar_at_typed::<T>(array, index, nullability)
        }))
    }
}

fn scalar_at_typed<T>(
    array: ArrayView<'_, Rans>,
    index: usize,
    nullability: vortex_array::dtype::Nullability,
) -> Scalar
where
    T: NativePType + Into<PValue>,
    u64: AsPrimitive<T>,
{
    let block_shift = array.block_shift();
    let block_values = 1usize << block_shift;
    let pos = usize::from(array.offset()) + index;
    let block_idx = pos >> block_shift;
    let pos_in_block = pos & (block_values - 1);
    let target_stripe = pos_in_block >> 5;
    let n_stripes = target_stripe + 1;

    let headers_bytes = array.block_headers().as_host();
    let payload_bytes = array.payload().as_host();
    let mut scratch: [MaybeUninit<T>; MAX_BLOCK_VALUES] =
        [const { MaybeUninit::uninit() }; MAX_BLOCK_VALUES];
    decode_block_stripes::<T>(
        array.decode_table(),
        headers_bytes.as_ref(),
        payload_bytes.as_ref(),
        array.payload_base_offset(),
        block_idx,
        n_stripes,
        array.has_extra_bits(),
        &mut scratch[..n_stripes * NUM_LANES],
    );
    // SAFETY: `pos_in_block < n_stripes * NUM_LANES`, and `decode_block_stripes` initialized all
    // `n_stripes * NUM_LANES` elements of `scratch`.
    let value = unsafe { scratch[pos_in_block].assume_init() };
    Scalar::primitive::<T>(value, nullability)
}

pub(crate) fn decompress_rans(
    array: ArrayView<'_, Rans>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let ptype = array.dtype().as_ptype();
    let validity = array.rans_validity();
    Ok(match_each_integer_ptype!(ptype, |T| {
        decompress_rans_mapped::<T, _>(array, validity, ctx, |u| u.as_())
    }))
}

pub(crate) fn decompress_rans_mapped<Out, F>(
    array: ArrayView<'_, Rans>,
    validity: Validity,
    ctx: &mut ExecutionCtx,
    map_out: F,
) -> PrimitiveArray
where
    Out: NativePType,
    F: Copy + Fn(u64) -> Out,
{
    let len = array.len();
    if len == 0 {
        return PrimitiveArray::new(BufferMut::<Out>::with_capacity(0).freeze(), validity);
    }

    let block_shift = array.block_shift();
    let block_values = 1usize << block_shift;
    let offset = usize::from(array.offset());
    let total_end = offset + len;
    let num_blocks = total_end.div_ceil(block_values);

    let headers_bytes = array.block_headers().as_host();
    let payload_bytes = array.payload().as_host();
    let table = array.decode_table();
    let base_offset = array.payload_base_offset();
    let has_extra_bits = array.has_extra_bits();

    let mut out = BufferMut::<Out>::with_capacity_in(len, ctx.allocator().clone());
    let out_uninit = &mut out.spare_capacity_mut()[..len];

    let mut scratch: [MaybeUninit<Out>; MAX_BLOCK_VALUES] =
        [const { MaybeUninit::uninit() }; MAX_BLOCK_VALUES];
    let mut written = 0usize;

    for block_idx in 0..num_blocks {
        let block_abs_start = block_idx << block_shift;
        let in_block_start = if block_idx == 0 { offset } else { 0 };
        let in_block_end = (total_end - block_abs_start).min(block_values);
        let copy_len = in_block_end - in_block_start;
        let n_stripes = in_block_end.div_ceil(NUM_LANES);

        if in_block_start == 0 && copy_len == block_values {
            decode_block_stripes_into(
                table,
                headers_bytes.as_ref(),
                payload_bytes.as_ref(),
                base_offset,
                block_idx,
                n_stripes,
                has_extra_bits,
                &mut out_uninit[written..written + block_values],
                map_out,
            );
        } else {
            decode_block_stripes_into(
                table,
                headers_bytes.as_ref(),
                payload_bytes.as_ref(),
                base_offset,
                block_idx,
                n_stripes,
                has_extra_bits,
                &mut scratch[..n_stripes * NUM_LANES],
                map_out,
            );
            out_uninit[written..written + copy_len]
                .copy_from_slice(&scratch[in_block_start..in_block_end]);
        }
        written += copy_len;
    }
    debug_assert_eq!(written, len);
    // SAFETY: Every element in `0..len` of `out` was initialized above.
    unsafe { out.set_len(len) };

    PrimitiveArray::new(out.freeze(), validity)
}

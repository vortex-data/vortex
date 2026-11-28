// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`Pivco`] array definition, [`PivcoData`] encapsulated operations, [`VTable`],
//! [`ValidityVTable`], and [`OperationsVTable`].

#![expect(clippy::cast_possible_truncation)]

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::ops::Range;
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
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::require_validity;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Alignment;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::registry::CachedId;

use crate::codec::BlockReader;
use crate::codec::DecodeWorkspace;
use crate::codec::MAX_BLOCK_SHIFT;
use crate::codec::MIN_BLOCK_SHIFT;
use crate::codec::PivcoBlockOffset;
use crate::codec::PivcoOptions;
use crate::codec::RankPredicate;
use crate::codec::block_header_stride;
use crate::codec::encode_primitive_slice;
use crate::codec::validate_block_headers;
use crate::pow2_bin::encode_pow2_bin;
use crate::pow2_bin::max_nonneg_u64;
use crate::pow2_bin::sym_max_value;
use crate::tree::PivcoTree;
use crate::tree::compute_tree_max_ebits;

/// The serialized and in-memory [`ArrayId`] of [`Pivco`] (`"vortex.pivco"`).
pub fn pivco_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.pivco");
    *ID
}

/// Serialized protobuf metadata for a [`PivcoArray`].
#[derive(Clone, PartialEq, Eq, prost::Message)]
pub struct PivcoMetadata {
    /// Element offset within the first block (`0..block_values`, or `0` when empty).
    #[prost(uint32, tag = "1")]
    pub offset: u32,
    /// Encoded element count of the final block (`1..=block_values`, or `0` when empty).
    #[prost(uint32, tag = "2")]
    pub last_block_len: u32,
    /// `log2(block_values)` in `10..=13` (`1,024..=8,192`); values outside this range (including `0`) are rejected on read.
    #[prost(uint32, tag = "3")]
    pub block_shift: u32,
    /// Base byte offset subtracted from `block_headers` `tree_byte_offset` when sliced.
    #[prost(uint32, tag = "4")]
    pub tree_base_offset: u32,
    /// Base `u64`-word offset subtracted from `block_headers` `raw_word_offset` when sliced.
    #[prost(uint32, tag = "5")]
    pub raw_base_word_offset: u32,
    /// Compact serialized `PivcoTree` pre-order opcodes.
    #[prost(bytes, tag = "6")]
    pub tree_bytes: Vec<u8>,
    /// Whether child 0 is a serialized validity bitmap array.
    #[prost(bool, tag = "7")]
    pub has_validity: bool,
}

/// A [`Pivco`]-encoded Vortex array.
pub type PivcoArray = Array<Pivco>;

/// Child slots of a [`PivcoArray`].
#[array_slots(Pivco)]
pub struct PivcoSlots {
    /// Optional validity bitmap child (`None` for `NonNullable` and `AllValid`).
    #[slot(0)]
    pub validity_child: Option<ArrayRef>,
}

/// Marker struct for the `vortex.pivco` encoding.
#[derive(Clone, Debug)]
pub struct Pivco;

/// Encoding-specific state and buffers for a [`PivcoArray`].
#[derive(Clone, Debug)]
pub struct PivcoData {
    block_shift: u8,
    offset: u16,
    last_block_len: u16,
    tree_base_offset: u32,
    raw_base_word_offset: u32,
    has_extra_bits: bool,
    max_ebits: u8,
    tree: Arc<PivcoTree>,
    tree_bytes: Arc<[u8]>,
    block_headers: BufferHandle,
    tree_payload: BufferHandle,
    raw_bits: BufferHandle,
}

/// Result of evaluating a direct spine-walking constant comparison on [`PivcoData`].
pub(crate) enum ConstantCompareResult {
    /// Constant boolean result for all valid elements.
    Constant(bool),
    /// Per-element packed bitmap result.
    Bits(BitBuffer),
}

/// Host-pinned reusable multi-block decoder for a [`PivcoData`] chunk.
pub(crate) struct PivcoBlockDecoder<'a> {
    reader: BlockReader<'a>,
    ws: DecodeWorkspace,
}

impl<'a> PivcoBlockDecoder<'a> {
    fn new(reader: BlockReader<'a>) -> Self {
        let ws = DecodeWorkspace::for_tree(reader.tree, reader.block_values());
        Self { reader, ws }
    }

    /// Decodes elements `[in_block_start..in_block_end)` of `block_idx` directly into `dst_slice`,
    /// applying `map_out` to each reconstructed `u64` value in register.
    #[inline]
    pub(crate) fn decode_range_into<Out, F>(
        &mut self,
        block_idx: usize,
        in_block_start: usize,
        in_block_end: usize,
        dst_slice: &mut [MaybeUninit<Out>],
        map_out: F,
    ) -> VortexResult<()>
    where
        Out: Copy,
        F: Copy + Fn(u64) -> Out,
    {
        self.reader.decode_range_into(
            block_idx,
            in_block_start,
            in_block_end,
            &mut self.ws,
            dst_slice,
            map_out,
        )
    }

    /// Decodes a single element `(block_idx, pos_in_block)` via top-down `O(depth)` point access.
    #[inline]
    pub(crate) fn decode_value_at(
        &self,
        block_idx: usize,
        pos_in_block: usize,
    ) -> VortexResult<u64> {
        self.reader.decode_value_at(block_idx, pos_in_block)
    }
}

impl Display for PivcoData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "order_preserving: {}, block_shift: {}, offset: {}, last_block_len: {}, extra_bits: {}, leaves: {}",
            self.tree.is_order_preserving,
            self.block_shift,
            self.offset,
            self.last_block_len,
            self.has_extra_bits,
            self.tree.rank_to_sym.len()
        )
    }
}

impl ArrayHash for PivcoData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.block_shift.hash(state);
        self.offset.hash(state);
        self.last_block_len.hash(state);
        self.tree_base_offset.hash(state);
        self.raw_base_word_offset.hash(state);
        self.has_extra_bits.hash(state);
        self.tree_bytes.hash(state);
        self.block_headers.array_hash(state, accuracy);
        self.tree_payload.array_hash(state, accuracy);
        self.raw_bits.array_hash(state, accuracy);
    }
}

impl ArrayEq for PivcoData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.block_shift == other.block_shift
            && self.offset == other.offset
            && self.last_block_len == other.last_block_len
            && self.tree_base_offset == other.tree_base_offset
            && self.raw_base_word_offset == other.raw_base_word_offset
            && self.has_extra_bits == other.has_extra_bits
            && self.tree_bytes == other.tree_bytes
            && self.block_headers.array_eq(&other.block_headers, accuracy)
            && self.tree_payload.array_eq(&other.tree_payload, accuracy)
            && self.raw_bits.array_eq(&other.raw_bits, accuracy)
    }
}

/// Number of `2^block_shift`-element blocks spanned by `offset + len` elements (`0` when `len == 0`),
/// rejecting `offset + len` overflow on malformed inputs.
#[inline]
pub(crate) fn checked_num_blocks_for(
    block_shift: u8,
    offset: u16,
    len: usize,
) -> VortexResult<usize> {
    if len == 0 {
        Ok(0)
    } else {
        let total = usize::from(offset)
            .checked_add(len)
            .ok_or_else(|| vortex_err!("Pivco offset ({offset}) + len ({len}) overflows usize"))?;
        Ok(total.div_ceil(1usize << block_shift))
    }
}

/// Number of `2^block_shift`-element blocks spanned by `offset + len` elements (`0` when `len == 0`).
#[inline]
pub(crate) fn num_blocks_for(block_shift: u8, offset: u16, len: usize) -> usize {
    if len == 0 {
        0
    } else {
        (usize::from(offset) + len).div_ceil(1usize << block_shift)
    }
}

impl PivcoData {
    /// Constructs [`PivcoData`] from validated components and a compiled [`PivcoTree`].
    ///
    /// `has_extra_bits`, `max_ebits`, and `tree_bytes` are derived directly from `tree` so
    /// in-memory state and wire metadata can never disagree.
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        block_shift: u8,
        offset: u16,
        last_block_len: u16,
        tree_base_offset: u32,
        raw_base_word_offset: u32,
        tree: Arc<PivcoTree>,
        block_headers: BufferHandle,
        tree_payload: BufferHandle,
        raw_bits: BufferHandle,
    ) -> VortexResult<Self> {
        let tree_bytes: Arc<[u8]> = Arc::from(tree.to_bytes());
        Self::try_new_with_cached_tree_bytes(
            block_shift,
            offset,
            last_block_len,
            tree_base_offset,
            raw_base_word_offset,
            tree,
            tree_bytes,
            block_headers,
            tree_payload,
            raw_bits,
        )
    }

    #[expect(clippy::too_many_arguments)]
    fn try_new_with_cached_tree_bytes(
        block_shift: u8,
        offset: u16,
        last_block_len: u16,
        tree_base_offset: u32,
        raw_base_word_offset: u32,
        tree: Arc<PivcoTree>,
        tree_bytes: Arc<[u8]>,
        block_headers: BufferHandle,
        tree_payload: BufferHandle,
        raw_bits: BufferHandle,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            (MIN_BLOCK_SHIFT..=MAX_BLOCK_SHIFT).contains(&block_shift),
            "Pivco block_shift {block_shift} must be in {MIN_BLOCK_SHIFT}..={MAX_BLOCK_SHIFT}"
        );
        let block_values = 1usize << block_shift;
        vortex_ensure!(
            usize::from(offset) < block_values,
            "Pivco offset must be < {block_values}, got {offset}"
        );
        vortex_ensure!(
            usize::from(last_block_len) <= block_values,
            "Pivco last_block_len must be <= {block_values}, got {last_block_len}"
        );
        let max_ebits = compute_tree_max_ebits(&tree);
        let has_extra_bits = max_ebits > 0;
        Ok(Self {
            block_shift,
            offset,
            last_block_len,
            tree_base_offset,
            raw_base_word_offset,
            has_extra_bits,
            max_ebits,
            tree,
            tree_bytes,
            block_headers,
            tree_payload,
            raw_bits,
        })
    }

    /// Returns `true` iff all three payload buffers (`block_headers`, `tree_payload`, `raw_bits`)
    /// are currently resident on host memory.
    #[inline]
    pub(crate) fn is_on_host(&self) -> bool {
        self.block_headers.is_on_host()
            && self.tree_payload.is_on_host()
            && self.raw_bits.is_on_host()
    }

    pub(crate) fn validate(
        &self,
        dtype: &DType,
        len: usize,
        validity: &Validity,
    ) -> VortexResult<()> {
        vortex_ensure!(
            dtype.is_int(),
            "PivcoArray requires an integer dtype, got {dtype}"
        );
        let max_val = max_nonneg_u64(dtype.as_ptype())
            .ok_or_else(|| vortex_err!("PivcoArray requires an integer dtype, got {dtype}"))?;
        for &sym in &self.tree.rank_to_sym {
            let sym_max = sym_max_value(sym);
            vortex_ensure!(
                sym_max <= max_val,
                "PivcoTree symbol {sym} (max value {sym_max}) exceeds {dtype} non-negative bound {max_val}"
            );
        }

        vortex_ensure_eq!(dtype.nullability(), validity.nullability());
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure_eq!(
                validity_len,
                len,
                "PivcoArray validity length must match array length"
            );
        }
        let block_values = 1usize << self.block_shift;
        vortex_ensure!(
            usize::from(self.offset) < block_values,
            "PivcoArray offset {} must be < {block_values}",
            self.offset
        );

        let expected_blocks = checked_num_blocks_for(self.block_shift, self.offset, len)?;
        if len == 0 {
            vortex_ensure_eq!(self.offset, 0, "Empty PivcoArray must have offset 0");
            vortex_ensure_eq!(
                self.last_block_len,
                0,
                "Empty PivcoArray must have last_block_len 0"
            );
            vortex_ensure_eq!(
                self.tree_base_offset,
                0,
                "Empty PivcoArray must have tree_base_offset 0"
            );
            vortex_ensure_eq!(
                self.raw_base_word_offset,
                0,
                "Empty PivcoArray must have raw_base_word_offset 0"
            );
            vortex_ensure!(
                self.block_headers.is_empty(),
                "Empty PivcoArray must have empty block_headers"
            );
            vortex_ensure!(
                self.tree_payload.is_empty(),
                "Empty PivcoArray must have empty tree_payload"
            );
            vortex_ensure!(
                self.raw_bits.is_empty(),
                "Empty PivcoArray must have empty raw_bits"
            );
            return Ok(());
        }

        vortex_ensure!(
            self.last_block_len >= 1 && usize::from(self.last_block_len) <= block_values,
            "PivcoArray last_block_len {} must be in 1..={block_values}",
            self.last_block_len
        );
        let total_end = usize::from(self.offset) + len;
        let last_block_start = (expected_blocks - 1) * block_values;
        vortex_ensure!(
            last_block_start < total_end
                && total_end <= last_block_start + usize::from(self.last_block_len),
            "PivcoArray offset ({}) + len ({len}) out of bounds for {expected_blocks} blocks with last_block_len {}",
            self.offset,
            self.last_block_len
        );

        let stride = block_header_stride(self.has_extra_bits);
        vortex_ensure_eq!(
            self.block_headers.len(),
            (expected_blocks + 1) * stride,
            "PivcoArray block_headers buffer size mismatch"
        );
        if self.block_headers.is_on_host() {
            vortex_ensure!(
                self.block_headers.is_aligned_to(Alignment::of::<u32>()),
                "PivcoArray block_headers buffer must be 4-byte aligned on host"
            );
        }
        if self.tree_payload.is_on_host() && !self.tree_payload.is_empty() {
            vortex_ensure!(
                self.tree_payload.is_aligned_to(Alignment::of::<u32>()),
                "PivcoArray tree_payload buffer must be 4-byte aligned on host"
            );
        }

        if !self.has_extra_bits {
            vortex_ensure_eq!(
                self.raw_base_word_offset,
                0,
                "PivcoArray raw_base_word_offset must be 0 when !has_extra_bits"
            );
            vortex_ensure!(
                self.raw_bits.is_empty(),
                "PivcoArray raw_bits must be empty when !has_extra_bits"
            );
        } else {
            vortex_ensure!(
                self.raw_bits.len().is_multiple_of(8),
                "PivcoArray raw_bits byte length {} must be a multiple of 8",
                self.raw_bits.len()
            );
            if self.raw_bits.is_on_host() && !self.raw_bits.is_empty() {
                vortex_ensure!(
                    self.raw_bits.is_aligned_to(Alignment::of::<u64>()),
                    "PivcoArray raw_bits buffer must be 8-byte aligned on host"
                );
            }
        }

        if self.block_headers.is_on_host() {
            validate_block_headers(
                self.block_headers.as_host().as_ref(),
                self.tree_payload.len(),
                self.raw_bits.len(),
                self.tree_base_offset,
                self.raw_base_word_offset,
                expected_blocks,
                self.has_extra_bits,
            )?;
        }
        Ok(())
    }

    pub(crate) fn buffer(&self, idx: usize) -> BufferHandle {
        match idx {
            0 => self.block_headers.clone(),
            1 => self.tree_payload.clone(),
            2 => self.raw_bits.clone(),
            _ => vortex_panic!("PivcoArray buffer index {idx} out of bounds"),
        }
    }

    pub(crate) fn with_buffers(&self, buffers: &[BufferHandle]) -> VortexResult<Self> {
        vortex_ensure_eq!(buffers.len(), 3);
        let mut data = self.clone();
        data.block_headers = buffers[0].clone();
        data.tree_payload = buffers[1].clone();
        data.raw_bits = buffers[2].clone();
        Ok(data)
    }

    pub(crate) fn to_metadata(&self, has_validity: bool) -> PivcoMetadata {
        PivcoMetadata {
            offset: u32::from(self.offset),
            last_block_len: u32::from(self.last_block_len),
            block_shift: u32::from(self.block_shift),
            tree_base_offset: self.tree_base_offset,
            raw_base_word_offset: self.raw_base_word_offset,
            tree_bytes: self.tree_bytes.to_vec(),
            has_validity,
        }
    }

    pub(crate) fn block_reader(&self, array_len: usize) -> VortexResult<BlockReader<'_>> {
        vortex_ensure!(
            self.is_on_host(),
            "PivcoArray buffers must be host-resident for CPU access"
        );
        Ok(BlockReader {
            tree: &self.tree,
            headers_bytes: self.block_headers.as_host().as_ref(),
            tree_payload: self.tree_payload.as_host().as_ref(),
            raw_bits: self.raw_bits.as_host().as_ref(),
            tree_base_offset: self.tree_base_offset,
            raw_base_word_offset: self.raw_base_word_offset,
            block_shift: self.block_shift,
            offset: self.offset,
            len: array_len,
            last_block_len: self.last_block_len,
            has_extra_bits: self.has_extra_bits,
            max_ebits: self.max_ebits,
        })
    }

    pub(crate) fn block_decoder(&self, array_len: usize) -> VortexResult<PivcoBlockDecoder<'_>> {
        Ok(PivcoBlockDecoder::new(self.block_reader(array_len)?))
    }

    pub(crate) fn decode_value_at_index(&self, len: usize, index: usize) -> VortexResult<u64> {
        let reader = self.block_reader(len)?;
        let pos = usize::from(self.offset) + index;
        let block_idx = pos >> self.block_shift;
        let pos_in_block = pos & (reader.block_values() - 1);
        reader.decode_value_at(block_idx, pos_in_block)
    }

    pub(crate) fn scalar_at(
        &self,
        dtype: &DType,
        len: usize,
        index: usize,
    ) -> VortexResult<Scalar> {
        let ptype = dtype.as_ptype();
        let nullability = dtype.nullability();
        match_each_integer_ptype!(ptype, |T| {
            self.scalar_at_typed::<T>(len, index, nullability)
        })
    }

    fn scalar_at_typed<T>(
        &self,
        len: usize,
        index: usize,
        nullability: Nullability,
    ) -> VortexResult<Scalar>
    where
        T: NativePType + Into<PValue>,
        u64: AsPrimitive<T>,
    {
        let raw_u64 = self.decode_value_at_index(len, index)?;
        Ok(Scalar::primitive::<T>(raw_u64.as_(), nullability))
    }

    pub(crate) fn decompress(
        &self,
        ptype: PType,
        len: usize,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        match_each_integer_ptype!(ptype, |T| {
            self.decompress_mapped::<T, _>(len, validity, ctx, |u| u.as_())
        })
    }

    pub(crate) fn decompress_mapped<Out, F>(
        &self,
        len: usize,
        validity: Validity,
        ctx: &mut ExecutionCtx,
        map_out: F,
    ) -> VortexResult<PrimitiveArray>
    where
        Out: NativePType,
        F: Copy + Fn(u64) -> Out,
    {
        if len == 0 {
            return Ok(PrimitiveArray::new(
                BufferMut::<Out>::with_capacity(0).freeze(),
                validity,
            ));
        }

        let block_values = 1usize << self.block_shift;
        let offset = usize::from(self.offset);
        let total_end = offset + len;
        let num_blocks = total_end.div_ceil(block_values);

        let mut out = BufferMut::<Out>::with_capacity_in(len, ctx.allocator().clone());
        let out_uninit = &mut out.spare_capacity_mut()[..len];

        let mut decoder = self.block_decoder(len)?;
        let mut written = 0usize;

        for block_idx in 0..num_blocks {
            let block_abs_start = block_idx * block_values;
            let in_block_start = if block_idx == 0 { offset } else { 0 };
            let in_block_end = (total_end - block_abs_start).min(block_values);
            let copy_len = in_block_end - in_block_start;

            decoder.decode_range_into(
                block_idx,
                in_block_start,
                in_block_end,
                &mut out_uninit[written..written + copy_len],
                map_out,
            )?;
            written += copy_len;
        }

        debug_assert_eq!(written, len);
        // SAFETY: Every element in `0..len` of `out` was initialized above.
        unsafe { out.set_len(len) };

        Ok(PrimitiveArray::new(out.freeze(), validity))
    }

    pub(crate) fn slice(&self, len: usize, range: Range<usize>) -> VortexResult<Self> {
        let block_shift = self.block_shift;
        let block_values = 1usize << block_shift;
        let has_extra_bits = self.has_extra_bits;

        if range.is_empty() {
            return Self::try_new_with_cached_tree_bytes(
                block_shift,
                0,
                0,
                0,
                0,
                Arc::clone(&self.tree),
                Arc::clone(&self.tree_bytes),
                self.block_headers.slice(0..0),
                self.tree_payload.slice(0..0),
                self.raw_bits.slice(0..0),
            );
        }

        let old_num_blocks = num_blocks_for(block_shift, self.offset, len);
        let abs_start = usize::from(self.offset) + range.start;
        let abs_stop = usize::from(self.offset) + range.end;
        let first_block = abs_start >> block_shift;
        let end_block = abs_stop.div_ceil(block_values);
        let new_offset = (abs_start & (block_values - 1)) as u16;
        let new_last_block_len = if end_block < old_num_blocks {
            block_values as u16
        } else {
            self.last_block_len
        };

        let stride = block_header_stride(has_extra_bits);
        let (new_headers, new_tree_payload, new_raw_bits, new_tree_base, new_raw_base) =
            if self.is_on_host() {
                let hdr_host = self.block_headers.as_host();
                let first_hdr =
                    PivcoBlockOffset::read_at(hdr_host.as_ref(), first_block, has_extra_bits);
                let end_hdr =
                    PivcoBlockOffset::read_at(hdr_host.as_ref(), end_block, has_extra_bits);

                let new_headers = BufferHandle::new_host(hdr_host.slice_with_alignment(
                    first_block * stride..(end_block + 1) * stride,
                    Alignment::of::<u32>(),
                ));

                let old_tree_base = self.tree_base_offset as usize;
                let rel_tree_start = (first_hdr.tree_byte_offset as usize) - old_tree_base;
                let rel_tree_end = (end_hdr.tree_byte_offset as usize) - old_tree_base;
                let new_tree_payload = BufferHandle::new_host(
                    self.tree_payload
                        .as_host()
                        .slice_with_alignment(rel_tree_start..rel_tree_end, Alignment::of::<u32>()),
                );
                let new_tree_base = first_hdr.tree_byte_offset;

                let (new_raw_bits, new_raw_base) = if has_extra_bits {
                    let old_raw_base = self.raw_base_word_offset as usize;
                    let rel_raw_start = ((first_hdr.raw_word_offset as usize) - old_raw_base) * 8;
                    let rel_raw_end = ((end_hdr.raw_word_offset as usize) - old_raw_base) * 8;
                    (
                        BufferHandle::new_host(self.raw_bits.as_host().slice_with_alignment(
                            rel_raw_start..rel_raw_end,
                            Alignment::of::<u64>(),
                        )),
                        first_hdr.raw_word_offset,
                    )
                } else {
                    (
                        BufferHandle::new_host(
                            ByteBufferMut::with_capacity_aligned(0, Alignment::of::<u64>())
                                .freeze(),
                        ),
                        0,
                    )
                };
                (
                    new_headers,
                    new_tree_payload,
                    new_raw_bits,
                    new_tree_base,
                    new_raw_base,
                )
            } else {
                // Zero-sync device `SliceReduce`: trim only `block_headers` in O(1) without
                // reading or copying device-resident `tree_payload` / `raw_bits`.
                (
                    self.block_headers
                        .slice(first_block * stride..(end_block + 1) * stride),
                    self.tree_payload.clone(),
                    self.raw_bits.clone(),
                    self.tree_base_offset,
                    self.raw_base_word_offset,
                )
            };

        Self::try_new_with_cached_tree_bytes(
            block_shift,
            new_offset,
            new_last_block_len,
            new_tree_base,
            new_raw_base,
            Arc::clone(&self.tree),
            Arc::clone(&self.tree_bytes),
            new_headers,
            new_tree_payload,
            new_raw_bits,
        )
    }

    pub(crate) fn compare_constant_u64(
        &self,
        len: usize,
        rhs_u64: u64,
        operator: CompareOperator,
    ) -> VortexResult<Option<ConstantCompareResult>> {
        if !self.is_on_host() {
            return Ok(None);
        }
        let tree = &self.tree;
        match operator {
            CompareOperator::Eq | CompareOperator::NotEq => {
                let negate = operator == CompareOperator::NotEq;
                let (sym, ebits, _) = encode_pow2_bin(rhs_u64);
                let rank = tree.sym_to_rank[usize::from(sym)];
                let sym_present = tree
                    .rank_to_sym
                    .get(usize::from(rank))
                    .is_some_and(|&s| s == sym);

                if !sym_present {
                    // Even for `rhs_u64 >= 32` (`ebits > 0`), if its `Pow2-Bin` bucket is completely
                    // absent from the chunk's tree, `Eq` is 100% `false` in `O(1)`.
                    return Ok(Some(ConstantCompareResult::Constant(negate)));
                }

                // Direct spine-walking equality when `ebits == 0` (`rhs_u64 < 32`).
                if ebits == 0 {
                    if tree.rank_to_sym.len() == 1 {
                        return Ok(Some(ConstantCompareResult::Constant(!negate)));
                    }
                    let reader = self.block_reader(len)?;
                    let bits = reader.evaluate_spine_predicate(RankPredicate::Eq(rank), negate)?;
                    return Ok(Some(ConstantCompareResult::Bits(bits)));
                }

                Ok(None)
            }
            CompareOperator::Lt
            | CompareOperator::Lte
            | CompareOperator::Gt
            | CompareOperator::Gte => {
                // Normalize all 4 ordering ops on `u >= 0` to `(u < threshold) ^ negate`.
                let (threshold, negate) = match operator {
                    CompareOperator::Lt => {
                        if rhs_u64 == 0 {
                            return Ok(Some(ConstantCompareResult::Constant(false)));
                        }
                        (rhs_u64, false)
                    }
                    CompareOperator::Gte => {
                        if rhs_u64 == 0 {
                            return Ok(Some(ConstantCompareResult::Constant(true)));
                        }
                        (rhs_u64, true)
                    }
                    CompareOperator::Lte => {
                        if rhs_u64 == u64::MAX {
                            return Ok(Some(ConstantCompareResult::Constant(true)));
                        }
                        (rhs_u64 + 1, false)
                    }
                    CompareOperator::Gt => {
                        if rhs_u64 == u64::MAX {
                            return Ok(Some(ConstantCompareResult::Constant(false)));
                        }
                        (rhs_u64 + 1, true)
                    }
                    _ => unreachable!(),
                };

                if !tree.is_order_preserving {
                    // Even in `Huffman` mode, `u < 1` (`u == 0`) or `u >= 1` (`u != 0`) is an exact
                    // `sym == 0` spine query.
                    if threshold == 1 {
                        let rank0 = tree.sym_to_rank[0];
                        let has_zero = tree
                            .rank_to_sym
                            .get(usize::from(rank0))
                            .is_some_and(|&s| s == 0);
                        if !has_zero {
                            return Ok(Some(ConstantCompareResult::Constant(negate)));
                        }
                        if tree.rank_to_sym.len() == 1 {
                            return Ok(Some(ConstantCompareResult::Constant(!negate)));
                        }
                        let reader = self.block_reader(len)?;
                        let bits =
                            reader.evaluate_spine_predicate(RankPredicate::Eq(rank0), negate)?;
                        return Ok(Some(ConstantCompareResult::Bits(bits)));
                    }
                    return Ok(None);
                }

                // In `OrderPreserving` mode (`rank_to_sym` strictly sorted):
                let (target_sym, _ebits, target_raw) = encode_pow2_bin(threshold);
                let split_rank = tree.rank_to_sym.partition_point(|&s| s < target_sym) as u8;
                let target_sym_present = tree
                    .rank_to_sym
                    .get(usize::from(split_rank))
                    .is_some_and(|&s| s == target_sym);

                // `u < threshold` is 100% decided by `rank < split_rank` whenever EITHER:
                // 1. `target_raw == 0` (every `threshold <= 32` AND every Pow2-Bin boundary), OR
                // 2. `target_sym` is absent from this chunk's active alphabet.
                if target_raw == 0 || !target_sym_present {
                    if split_rank == 0 {
                        return Ok(Some(ConstantCompareResult::Constant(negate)));
                    }
                    if usize::from(split_rank) == tree.rank_to_sym.len() {
                        return Ok(Some(ConstantCompareResult::Constant(!negate)));
                    }
                    let reader = self.block_reader(len)?;
                    let bits =
                        reader.evaluate_spine_predicate(RankPredicate::Lt(split_rank), negate)?;
                    return Ok(Some(ConstantCompareResult::Bits(bits)));
                }

                Ok(None)
            }
        }
    }

    /// `log2(block_values)` in `10..=13`.
    #[inline]
    pub fn block_shift(&self) -> u8 {
        self.block_shift
    }

    /// Number of elements per full block (`1 << block_shift`).
    #[inline]
    pub fn block_values(&self) -> usize {
        1usize << self.block_shift
    }

    /// Element offset (`0..block_values`) within the first block.
    #[inline]
    pub fn offset(&self) -> u16 {
        self.offset
    }

    /// Encoded element count of the final block (`1..=block_values`, or `0` when empty).
    #[inline]
    pub fn last_block_len(&self) -> u16 {
        self.last_block_len
    }

    /// Base byte offset subtracted from `block_headers` `tree_byte_offset` into `tree_payload`.
    #[inline]
    pub fn tree_base_offset(&self) -> u32 {
        self.tree_base_offset
    }

    /// Base `u64`-word offset subtracted from `block_headers` `raw_word_offset` into `raw_bits`.
    #[inline]
    pub fn raw_base_word_offset(&self) -> u32 {
        self.raw_base_word_offset
    }

    /// Whether any active symbol in the chunk has `ebits > 0`.
    #[inline]
    pub fn has_extra_bits(&self) -> bool {
        self.has_extra_bits
    }

    /// Maximum `ebits` across all active symbols (`0..=62`).
    #[inline]
    pub fn max_ebits(&self) -> u8 {
        self.max_ebits
    }

    /// Whether the chunk's wavelet tree is order-preserving (supports direct range comparisons).
    #[inline]
    pub fn is_order_preserving(&self) -> bool {
        self.tree.is_order_preserving()
    }

    /// Shared chunk-level wavelet tree.
    #[cfg(test)]
    #[inline]
    pub(crate) fn tree(&self) -> &Arc<PivcoTree> {
        &self.tree
    }

    /// Serialized `PivcoTree` metadata bytes.
    #[inline]
    pub fn tree_bytes(&self) -> &Arc<[u8]> {
        &self.tree_bytes
    }

    /// Sentinel offset table buffer (`(num_blocks + 1)` entries of `4` or `8` bytes).
    #[inline]
    pub fn block_headers(&self) -> &BufferHandle {
        &self.block_headers
    }

    /// Contiguous byte-packed pre-order wavelet-tree node streams buffer.
    #[inline]
    pub fn tree_payload(&self) -> &BufferHandle {
        &self.tree_payload
    }

    /// Contiguous `u64`-word verbatim extra-bits buffer (`0` bytes when `!has_extra_bits`).
    #[inline]
    pub fn raw_bits(&self) -> &BufferHandle {
        &self.raw_bits
    }
}

/// Typed extension accessors for [`PivcoArray`] and `ArrayView<'_, Pivco>`.
pub trait PivcoArrayExt: TypedArrayRef<Pivco> {
    #[inline]
    fn block_shift(&self) -> u8 {
        PivcoData::block_shift(self)
    }

    #[inline]
    fn block_values(&self) -> usize {
        PivcoData::block_values(self)
    }

    #[inline]
    fn offset(&self) -> u16 {
        PivcoData::offset(self)
    }

    #[inline]
    fn last_block_len(&self) -> u16 {
        PivcoData::last_block_len(self)
    }

    #[inline]
    fn tree_base_offset(&self) -> u32 {
        PivcoData::tree_base_offset(self)
    }

    #[inline]
    fn raw_base_word_offset(&self) -> u32 {
        PivcoData::raw_base_word_offset(self)
    }

    #[inline]
    fn has_extra_bits(&self) -> bool {
        PivcoData::has_extra_bits(self)
    }

    #[inline]
    fn max_ebits(&self) -> u8 {
        PivcoData::max_ebits(self)
    }

    #[inline]
    fn is_order_preserving(&self) -> bool {
        PivcoData::is_order_preserving(self)
    }

    #[inline]
    fn tree_bytes(&self) -> &Arc<[u8]> {
        PivcoData::tree_bytes(self)
    }

    #[inline]
    fn block_headers(&self) -> &BufferHandle {
        PivcoData::block_headers(self)
    }

    #[inline]
    fn tree_payload(&self) -> &BufferHandle {
        PivcoData::tree_payload(self)
    }

    #[inline]
    fn raw_bits(&self) -> &BufferHandle {
        PivcoData::raw_bits(self)
    }

    #[inline]
    fn pivco_validity(&self) -> Validity {
        child_to_validity(self.validity_child(), self.dtype().nullability())
    }
}

impl<T: TypedArrayRef<Pivco>> PivcoArrayExt for T {}

impl Pivco {
    /// Constructs a validated [`PivcoArray`] from its parts.
    pub fn try_new(
        dtype: DType,
        len: usize,
        data: PivcoData,
        validity: Validity,
    ) -> VortexResult<PivcoArray> {
        let slots = PivcoSlots {
            validity_child: validity_to_child(&validity, len),
        }
        .into_slots();
        Array::try_from_parts(ArrayParts::new(Pivco, dtype, len, data, slots))
    }

    /// Encodes a non-negative integer [`PrimitiveArray`] using [`PivcoOptions::default()`] (`PHA`).
    pub fn encode(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<PivcoArray> {
        Self::encode_with_options(array, PivcoOptions::default(), ctx)
    }

    /// Encodes a non-negative integer [`PrimitiveArray`] with explicit [`PivcoOptions`].
    pub fn encode_with_options(
        array: &PrimitiveArray,
        opts: PivcoOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PivcoArray> {
        let dtype = array.dtype().clone();
        vortex_ensure!(
            dtype.is_int(),
            "Pivco::encode requires an integer PrimitiveArray, got {dtype}"
        );
        let ptype = array.ptype();
        let len = array.len();
        let validity = array.validity()?;
        let mask = validity.execute_mask(len, ctx)?;

        let encoded = match_each_integer_ptype!(ptype, |T| {
            encode_primitive_slice::<T>(array.as_slice::<T>(), ptype.is_signed_int(), &mask, opts)?
        });

        let data = PivcoData::try_new(
            encoded.block_shift,
            0,
            encoded.last_block_len,
            0,
            0,
            Arc::new(encoded.tree),
            BufferHandle::new_host(encoded.block_headers),
            BufferHandle::new_host(encoded.tree_payload),
            BufferHandle::new_host(encoded.raw_bits),
        )?;
        let out = Self::try_new(dtype, len, data, validity)?;
        out.statistics().inherit_from(array.statistics());
        Ok(out)
    }
}

impl VTable for Pivco {
    type TypedArrayData = PivcoData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        pivco_id()
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), PivcoSlots::COUNT);
        let validity = child_to_validity(
            PivcoSlotsView::from_slots(slots).validity_child,
            dtype.nullability(),
        );
        data.validate(dtype, len, &validity)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        3
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        array.data().buffer(idx)
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("block_headers".to_string()),
            1 => Some("tree_payload".to_string()),
            2 => Some("raw_bits".to_string()),
            _ => None,
        }
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        let data = array.data().with_buffers(buffers)?;
        Ok(ArrayParts::new(
            self.clone(),
            array.dtype().clone(),
            array.len(),
            data,
            array.slots().iter().cloned().collect(),
        ))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        PivcoSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        require_validity!(array, PivcoSlots::VALIDITY_CHILD);
        let view = array.as_view();
        let validity = Self::validity(view)?;
        let decoded = view
            .data()
            .decompress(view.dtype().as_ptype(), view.len(), validity, ctx)?;
        Ok(ExecutionResult::done(decoded.into_array()))
    }
}

impl ValidityVTable<Pivco> for Pivco {
    fn validity(array: ArrayView<'_, Pivco>) -> VortexResult<Validity> {
        Ok(child_to_validity(
            array.slots()[PivcoSlots::VALIDITY_CHILD].as_ref(),
            array.dtype().nullability(),
        ))
    }
}

impl OperationsVTable<Pivco> for Pivco {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Pivco>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array.data().scalar_at(array.dtype(), array.len(), index)
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
pub(crate) mod is_constant;
pub(crate) mod is_sorted;
mod list_contains;

use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use fastlanes::FoR as FastLanesFoR;
use num_traits::AsPrimitive;
use num_traits::WrappingAdd;
use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::arrays::filter::FilterReduce;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::filter::ChunkDecoder;
use vortex_buffer::filter::FILTER_CHUNK_LEN;
use vortex_buffer::filter::filter_chunked;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::FoRReferences;
use crate::for_::array::FoRArrayExt;
use crate::for_::array::FoRArraySlotsExt;

const _: () = assert!(FL_CHUNK_SIZE == FILTER_CHUNK_LEN);

impl TakeExecute for FoR {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        match array.references() {
            FoRReferences::Global(reference) => Ok(Some(
                FoR::try_new(array.encoded().take(indices.clone())?, reference.clone())?
                    .into_array(),
            )),
            FoRReferences::Blocked(references) => {
                take_blocked(array, references, indices, ctx).map(|arr| Some(arr.into_array()))
            }
        }
    }
}

impl FilterReduce for FoR {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let FoRReferences::Global(reference) = array.references() else {
            return Ok(None);
        };
        FoR::try_new(array.encoded().filter(mask.clone())?, reference.clone())
            .map(|a| Some(a.into_array()))
    }
}

impl FilterKernel for FoR {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        match array.references() {
            FoRReferences::Global(reference) => Ok(Some(
                FoR::try_new(array.encoded().filter(mask.clone())?, reference.clone())?
                    .into_array(),
            )),
            FoRReferences::Blocked(references) => {
                let Mask::Values(mask_values) = mask else {
                    return Ok(None);
                };
                filter_blocked(array, references, mask, mask_values, ctx)
                    .map(|arr| Some(arr.into_array()))
            }
        }
    }
}

fn filter_blocked(
    array: ArrayView<'_, FoR>,
    references: &ArrayRef,
    mask: &Mask,
    mask_values: &MaskValues,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let refs = references.clone().execute::<PrimitiveArray>(ctx)?;
    let offset = usize::from(array.offset());

    if let Some(bp) = array.encoded().as_opt::<BitPacked>()
        && bp.offset() == array.offset()
    {
        let validity = bp.validity()?.filter(mask)?;
        return match_each_integer_ptype!(refs.ptype(), |T| {
            filter_blocked_bitpacked::<T>(
                bp,
                refs.as_slice::<T>(),
                offset,
                validity,
                mask_values,
                ctx,
            )
        });
    }

    let filtered_encoded = array
        .encoded()
        .filter(mask.clone())?
        .execute::<PrimitiveArray>(ctx)?;

    if filtered_encoded.ptype() == refs.ptype() {
        match_each_integer_ptype!(refs.ptype(), |T| {
            filter_blocked_same_width::<T>(
                filtered_encoded,
                refs.as_slice::<T>(),
                offset,
                mask_values,
                ctx,
            )
        })
    } else {
        match_each_integer_ptype!(refs.ptype(), |T| {
            filter_blocked_narrow_t::<T>(
                filtered_encoded,
                refs.as_slice::<T>(),
                offset,
                mask_values,
                ctx,
            )
        })
    }
}

/// Fused FastLanes `BitPacked` + blocked-`FoR` chunk decoder for same-width types.
///
/// Decodes in `T::Physical` (unsigned two's-complement wrapping addition is bit-for-bit identical
/// to signed wrapping addition) so `filter_chunked` compacts chunks straight out of L1 cache.
struct FoRPackedSameWidth<'a, T: PhysicalPType> {
    packed: &'a [T::Physical],
    bit_width: usize,
    packed_chunk_len: usize,
    refs: &'a [T],
}

impl<'a, T: PhysicalPType> FoRPackedSameWidth<'a, T> {
    fn new(packed: &'a [T::Physical], bit_width: usize, refs: &'a [T]) -> Self {
        Self {
            packed,
            bit_width,
            packed_chunk_len: FL_CHUNK_SIZE * bit_width / (8 * size_of::<T::Physical>()),
            refs,
        }
    }

    #[inline]
    fn chunk(&self, chunk_idx: usize) -> &[T::Physical] {
        &self.packed[chunk_idx * self.packed_chunk_len..][..self.packed_chunk_len]
    }
}

// SAFETY: both `decode_chunk` and `decode_indices` initialize every element of `dst`.
unsafe impl<T> ChunkDecoder<T::Physical> for FoRPackedSameWidth<'_, T>
where
    T: PhysicalPType + AsPrimitive<T::Physical>,
    T::Physical: BitPacking + FastLanesFoR + WrappingAdd + Zero,
{
    #[inline]
    fn decode_chunk(
        &self,
        chunk_idx: usize,
        dst: &mut [MaybeUninit<T::Physical>; FILTER_CHUNK_LEN],
    ) {
        let ref_p: T::Physical = self.refs[chunk_idx].as_();
        if self.bit_width == 0 {
            dst.fill(MaybeUninit::new(ref_p));
            return;
        }
        let mut scratch = [T::Physical::zero(); FILTER_CHUNK_LEN];
        // SAFETY: `chunk` holds `FL_CHUNK_SIZE` packed values at `bit_width` and `scratch` holds
        // one chunk.
        unsafe {
            FastLanesFoR::unchecked_unfor_pack(
                self.bit_width,
                self.chunk(chunk_idx),
                ref_p,
                &mut scratch,
            );
        }
        for (out, &val) in dst.iter_mut().zip(&scratch) {
            out.write(val);
        }
    }

    #[inline]
    fn decode_indices(
        &self,
        chunk_idx: usize,
        indices: &[usize],
        dst: &mut [MaybeUninit<T::Physical>],
    ) {
        let ref_p: T::Physical = self.refs[chunk_idx].as_();
        if self.bit_width == 0 {
            dst.fill(MaybeUninit::new(ref_p));
            return;
        }
        debug_assert!(indices.iter().all(|&index| index < FL_CHUNK_SIZE));
        // SAFETY: every index is within the chunk and `unchecked_unpack_indices` initializes `dst`.
        unsafe {
            BitPacking::unchecked_unpack_indices(
                self.bit_width,
                self.chunk(chunk_idx),
                indices,
                dst,
            );
            for val in dst.assume_init_mut() {
                *val = val.wrapping_add(&ref_p);
            }
        }
    }
}

/// Fused FastLanes `BitPacked` + blocked-`FoR` chunk decoder for narrowed unsigned residuals.
struct FoRPackedNarrow<'a, T, U> {
    packed: &'a [U],
    bit_width: usize,
    packed_chunk_len: usize,
    refs: &'a [T],
}

impl<'a, T, U> FoRPackedNarrow<'a, T, U> {
    fn new(packed: &'a [U], bit_width: usize, refs: &'a [T]) -> Self {
        Self {
            packed,
            bit_width,
            packed_chunk_len: FL_CHUNK_SIZE * bit_width / (8 * size_of::<U>()),
            refs,
        }
    }

    #[inline]
    fn chunk(&self, chunk_idx: usize) -> &[U] {
        &self.packed[chunk_idx * self.packed_chunk_len..][..self.packed_chunk_len]
    }
}

// SAFETY: both `decode_chunk` and `decode_indices` initialize every element of `dst`.
unsafe impl<T, U> ChunkDecoder<T> for FoRPackedNarrow<'_, T, U>
where
    T: NativePType + WrappingAdd,
    U: BitPacking + Zero + Copy + AsPrimitive<T>,
{
    #[inline]
    fn decode_chunk(&self, chunk_idx: usize, dst: &mut [MaybeUninit<T>; FILTER_CHUNK_LEN]) {
        let ref_b = self.refs[chunk_idx];
        if self.bit_width == 0 {
            dst.fill(MaybeUninit::new(ref_b));
            return;
        }
        let mut scratch = [U::zero(); FILTER_CHUNK_LEN];
        // SAFETY: `chunk` holds `FL_CHUNK_SIZE` values at `bit_width` and `scratch` holds one chunk.
        unsafe {
            BitPacking::unchecked_unpack(self.bit_width, self.chunk(chunk_idx), &mut scratch)
        };
        for (out, &val) in dst.iter_mut().zip(&scratch) {
            out.write(ref_b.wrapping_add(&val.as_()));
        }
    }

    #[inline]
    fn decode_indices(&self, chunk_idx: usize, indices: &[usize], dst: &mut [MaybeUninit<T>]) {
        let ref_b = self.refs[chunk_idx];
        if self.bit_width == 0 {
            dst.fill(MaybeUninit::new(ref_b));
            return;
        }
        debug_assert!(indices.iter().all(|&index| index < FL_CHUNK_SIZE));
        let mut scratch = [const { MaybeUninit::<U>::uninit() }; FILTER_CHUNK_LEN];
        let n = indices.len();
        // SAFETY: every index is within the chunk and `unchecked_unpack_indices` initializes
        // `scratch[..n]`.
        unsafe {
            BitPacking::unchecked_unpack_indices(
                self.bit_width,
                self.chunk(chunk_idx),
                indices,
                &mut scratch[..n],
            );
        }
        for (out, raw) in dst.iter_mut().zip(&scratch[..n]) {
            // SAFETY: initialized by `unchecked_unpack_indices` above.
            let val = unsafe { raw.assume_init() };
            out.write(ref_b.wrapping_add(&val.as_()));
        }
    }
}

fn filter_blocked_bitpacked<T>(
    bp: ArrayView<'_, BitPacked>,
    references: &[T],
    offset: usize,
    validity: Validity,
    mask_values: &MaskValues,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: IntegerPType + PhysicalPType + WrappingAdd + AsPrimitive<T> + AsPrimitive<T::Physical>,
    T::Physical: BitPacking + FastLanesFoR + NativePType + WrappingAdd,
    u8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    if bp.dtype().as_ptype() == T::PTYPE {
        let chunks = FoRPackedSameWidth::<T>::new(
            bp.packed_slice::<T::Physical>(),
            usize::from(bp.bit_width()),
            references,
        );
        let buffer = filter_chunked(&chunks, offset, mask_values.selection(), ctx.allocator());
        let buffer = if let Some(patches) = bp.patches() {
            apply_filtered_patches_buffer::<T, T, T::Physical>(
                buffer,
                &patches,
                offset,
                references,
                mask_values,
                ctx,
            )?
        } else {
            buffer
        };
        Ok(PrimitiveArray::from_byte_buffer(
            buffer.into_byte_buffer(),
            T::PTYPE,
            validity,
        ))
    } else {
        match_each_unsigned_integer_ptype!(bp.dtype().as_ptype(), |U| {
            filter_blocked_bitpacked_narrow::<T, U>(
                bp,
                references,
                offset,
                validity,
                mask_values,
                ctx,
            )
        })
    }
}

fn filter_blocked_bitpacked_narrow<T, U>(
    bp: ArrayView<'_, BitPacked>,
    references: &[T],
    offset: usize,
    validity: Validity,
    mask_values: &MaskValues,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: NativePType + WrappingAdd + AsPrimitive<T>,
    U: PhysicalPType<Physical = U> + BitPacking + Zero + AsPrimitive<T>,
{
    let chunks = FoRPackedNarrow::<T, U>::new(
        bp.packed_slice::<U>(),
        usize::from(bp.bit_width()),
        references,
    );
    let buffer = filter_chunked(&chunks, offset, mask_values.selection(), ctx.allocator());
    let buffer = if let Some(patches) = bp.patches() {
        apply_filtered_patches_buffer::<T, U, T>(
            buffer,
            &patches,
            offset,
            references,
            mask_values,
            ctx,
        )?
    } else {
        buffer
    };
    Ok(PrimitiveArray::new(buffer, validity))
}

fn apply_filtered_patches_buffer<T, U, O>(
    buffer: Buffer<O>,
    patches: &Patches,
    for_offset: usize,
    references: &[T],
    mask_values: &MaskValues,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<O>>
where
    T: NativePType + WrappingAdd + AsPrimitive<O>,
    U: NativePType + AsPrimitive<T>,
    O: NativePType,
{
    let p_idx = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let p_val = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
    let p_off = patches.offset();

    let mut out = match buffer.try_into_mut() {
        Ok(mut_buf) => mut_buf,
        Err(shared) => {
            let mut copy = BufferMut::<O>::with_capacity_in(shared.len(), ctx.allocator().clone());
            copy.extend_from_slice(shared.as_slice());
            copy
        }
    };

    match_each_unsigned_integer_ptype!(p_idx.ptype(), |I| {
        apply_filtered_patches_typed::<T, U, O, I>(
            out.as_mut_slice(),
            p_idx.as_slice::<I>(),
            p_val.as_slice::<U>(),
            p_off,
            for_offset,
            references,
            mask_values,
        )
    });

    Ok(out.freeze())
}

/// Galloping advance in sorted `indices` to the first element `>= target`.
#[inline]
fn gallop_ge<I: AsPrimitive<usize>>(indices: &[I], start: usize, target: usize) -> usize {
    let len = indices.len();
    if start >= len || indices[start].as_() >= target {
        return start;
    }
    let mut lo = start;
    let mut step = 1usize;
    while lo + step < len && indices[lo + step].as_() < target {
        lo += step;
        step <<= 1;
    }
    let hi = (lo + step).min(len);
    lo + 1 + indices[lo + 1..hi].partition_point(|&i| i.as_() < target)
}

fn apply_filtered_patches_typed<T, U, O, I>(
    out: &mut [O],
    p_idx: &[I],
    p_val: &[U],
    p_off: usize,
    for_offset: usize,
    references: &[T],
    mask_values: &MaskValues,
) where
    T: NativePType + WrappingAdd + AsPrimitive<O>,
    U: Copy + AsPrimitive<T>,
    O: Copy + 'static,
    I: AsPrimitive<usize>,
{
    let true_count = mask_values.true_count();
    if p_idx.is_empty() || true_count == 0 {
        return;
    }

    if p_idx.len() > true_count.saturating_mul(16) {
        let mut cursor = 0usize;
        if let Some(indices) = mask_values.cached_indices() {
            for (out_rank, &orig_pos) in indices.iter().enumerate() {
                let target = orig_pos + p_off;
                cursor = gallop_ge(p_idx, cursor, target);
                if cursor < p_idx.len() && p_idx[cursor].as_() == target {
                    let ref_b = references[(for_offset + orig_pos) / FL_CHUNK_SIZE];
                    out[out_rank] = ref_b.wrapping_add(&p_val[cursor].as_()).as_();
                    cursor += 1;
                }
            }
        } else {
            for (out_rank, orig_pos) in mask_values.bit_buffer().set_indices().enumerate() {
                let target = orig_pos + p_off;
                cursor = gallop_ge(p_idx, cursor, target);
                if cursor < p_idx.len() && p_idx[cursor].as_() == target {
                    let ref_b = references[(for_offset + orig_pos) / FL_CHUNK_SIZE];
                    out[out_rank] = ref_b.wrapping_add(&p_val[cursor].as_()).as_();
                    cursor += 1;
                }
            }
        }
        return;
    }

    let mask_bits = mask_values.bit_buffer();
    let mask_len = mask_bits.len();
    let mut prev_pos = 0usize;
    let mut rank = 0usize;
    for (&idx, &val) in p_idx.iter().zip(p_val) {
        let Some(orig_pos) = idx.as_().checked_sub(p_off) else {
            continue;
        };
        if orig_pos >= mask_len || !mask_bits.value(orig_pos) {
            continue;
        }
        rank += mask_bits.slice(prev_pos..orig_pos).true_count();
        prev_pos = orig_pos;
        let ref_b = references[(for_offset + orig_pos) / FL_CHUNK_SIZE];
        out[rank] = ref_b.wrapping_add(&val.as_()).as_();
    }
}

fn filter_blocked_same_width<T: NativePType + WrappingAdd>(
    encoded: PrimitiveArray,
    references: &[T],
    offset: usize,
    mask_values: &MaskValues,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let validity = encoded.validity()?;
    let len = encoded.len();
    let buffer = match encoded.into_buffer::<T>().try_into_mut() {
        Ok(mut values) => {
            if let Some(slices) = mask_values.cached_slices() {
                let mut out_idx = 0usize;
                for &(start, end) in slices {
                    let mut pos = start;
                    while pos < end {
                        let abs_pos = offset + pos;
                        let chunk = abs_pos / FL_CHUNK_SIZE;
                        let run_len = (end - pos).min(FL_CHUNK_SIZE - (abs_pos % FL_CHUNK_SIZE));
                        let reference = references[chunk];
                        for val in &mut values[out_idx..out_idx + run_len] {
                            *val = val.wrapping_add(&reference);
                        }
                        pos += run_len;
                        out_idx += run_len;
                    }
                }
                debug_assert_eq!(out_idx, len);
            } else {
                for (val, orig_pos) in values
                    .iter_mut()
                    .zip(mask_values.bit_buffer().set_indices())
                {
                    let reference = references[(offset + orig_pos) / FL_CHUNK_SIZE];
                    *val = val.wrapping_add(&reference);
                }
            }
            values.freeze()
        }
        Err(values) => {
            let mut out = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
            let dst = &mut out.spare_capacity_mut()[..len];
            let src = values.as_slice();
            if let Some(slices) = mask_values.cached_slices() {
                let mut out_idx = 0usize;
                for &(start, end) in slices {
                    let mut pos = start;
                    while pos < end {
                        let abs_pos = offset + pos;
                        let chunk = abs_pos / FL_CHUNK_SIZE;
                        let run_len = (end - pos).min(FL_CHUNK_SIZE - (abs_pos % FL_CHUNK_SIZE));
                        let reference = references[chunk];
                        for (slot, &val) in dst[out_idx..out_idx + run_len]
                            .iter_mut()
                            .zip(&src[out_idx..out_idx + run_len])
                        {
                            slot.write(val.wrapping_add(&reference));
                        }
                        pos += run_len;
                        out_idx += run_len;
                    }
                }
                debug_assert_eq!(out_idx, len);
            } else {
                for ((slot, &val), orig_pos) in dst
                    .iter_mut()
                    .zip(src)
                    .zip(mask_values.bit_buffer().set_indices())
                {
                    let reference = references[(offset + orig_pos) / FL_CHUNK_SIZE];
                    slot.write(val.wrapping_add(&reference));
                }
            }
            // SAFETY: every slot in `0..len` was initialized above.
            unsafe { out.set_len(len) };
            out.freeze()
        }
    };
    Ok(PrimitiveArray::new(buffer, validity))
}

fn filter_blocked_narrow_t<T>(
    encoded: PrimitiveArray,
    references: &[T],
    offset: usize,
    mask_values: &MaskValues,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: NativePType + WrappingAdd,
    u8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    match_each_unsigned_integer_ptype!(encoded.ptype(), |U| {
        filter_blocked_narrow::<T, U>(encoded, references, offset, mask_values, ctx)
    })
}

fn filter_blocked_narrow<T, U>(
    encoded: PrimitiveArray,
    references: &[T],
    offset: usize,
    mask_values: &MaskValues,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: NativePType + WrappingAdd,
    U: NativePType + AsPrimitive<T>,
{
    let validity = encoded.validity()?;
    let len = encoded.len();
    let src = encoded.as_slice::<U>();
    let mut out = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
    let dst = &mut out.spare_capacity_mut()[..len];
    if let Some(slices) = mask_values.cached_slices() {
        let mut out_idx = 0usize;
        for &(start, end) in slices {
            let mut pos = start;
            while pos < end {
                let abs_pos = offset + pos;
                let chunk = abs_pos / FL_CHUNK_SIZE;
                let run_len = (end - pos).min(FL_CHUNK_SIZE - (abs_pos % FL_CHUNK_SIZE));
                let reference = references[chunk];
                for (slot, &val) in dst[out_idx..out_idx + run_len]
                    .iter_mut()
                    .zip(&src[out_idx..out_idx + run_len])
                {
                    slot.write(reference.wrapping_add(&val.as_()));
                }
                pos += run_len;
                out_idx += run_len;
            }
        }
        debug_assert_eq!(out_idx, len);
    } else {
        for ((slot, &val), orig_pos) in dst
            .iter_mut()
            .zip(src)
            .zip(mask_values.bit_buffer().set_indices())
        {
            let reference = references[(offset + orig_pos) / FL_CHUNK_SIZE];
            slot.write(reference.wrapping_add(&val.as_()));
        }
    }
    // SAFETY: every slot in `0..len` was initialized above.
    unsafe { out.set_len(len) };
    Ok(PrimitiveArray::new(out.freeze(), validity))
}

fn take_blocked(
    array: ArrayView<'_, FoR>,
    references: &ArrayRef,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let taken_encoded = array
        .encoded()
        .take(indices.clone())?
        .execute::<PrimitiveArray>(ctx)?;
    let refs = references.clone().execute::<PrimitiveArray>(ctx)?;
    let indices_prim = indices.clone().execute::<PrimitiveArray>(ctx)?;
    let idx_mask = indices_prim
        .validity()?
        .execute_mask(indices_prim.len(), ctx)?;
    let offset = usize::from(array.offset());

    if taken_encoded.ptype() == refs.ptype() {
        match_each_integer_ptype!(refs.ptype(), |T| {
            take_blocked_same_width_t::<T>(
                taken_encoded,
                refs.as_slice::<T>(),
                &indices_prim,
                &idx_mask,
                offset,
                ctx,
            )
        })
    } else {
        match_each_integer_ptype!(refs.ptype(), |T| {
            take_blocked_narrow_t::<T>(
                taken_encoded,
                refs.as_slice::<T>(),
                &indices_prim,
                &idx_mask,
                offset,
                ctx,
            )
        })
    }
}

fn take_blocked_same_width_t<T: NativePType + WrappingAdd>(
    taken_encoded: PrimitiveArray,
    references: &[T],
    indices: &PrimitiveArray,
    idx_mask: &Mask,
    offset: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    match_each_integer_ptype!(indices.ptype(), |I| {
        take_blocked_same_width::<T, I>(
            taken_encoded,
            references,
            indices.as_slice::<I>(),
            idx_mask,
            offset,
            ctx,
        )
    })
}

fn take_blocked_same_width<T, I>(
    taken_encoded: PrimitiveArray,
    references: &[T],
    indices: &[I],
    idx_mask: &Mask,
    offset: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: NativePType + WrappingAdd,
    I: NativePType + AsPrimitive<usize>,
{
    let validity = taken_encoded.validity()?;
    let len = taken_encoded.len();
    if len == 0 || matches!(idx_mask, Mask::AllFalse(_)) {
        return Ok(taken_encoded);
    }

    let buffer = match taken_encoded.into_buffer::<T>().try_into_mut() {
        Ok(mut values) => {
            match idx_mask {
                Mask::AllTrue(_) => {
                    for (val, &idx) in values.as_mut_slice().iter_mut().zip(indices) {
                        let chunk = (offset + idx.as_()) / FL_CHUNK_SIZE;
                        *val = val.wrapping_add(&references[chunk]);
                    }
                }
                Mask::Values(mask_values) => {
                    let bits = mask_values.bit_buffer();
                    for (i, (val, &idx)) in
                        values.as_mut_slice().iter_mut().zip(indices).enumerate()
                    {
                        if bits.value(i) {
                            let chunk = (offset + idx.as_()) / FL_CHUNK_SIZE;
                            *val = val.wrapping_add(&references[chunk]);
                        }
                    }
                }
                Mask::AllFalse(_) => unreachable!(),
            }
            values.freeze()
        }
        Err(values) => {
            let mut out = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
            let dst = &mut out.spare_capacity_mut()[..len];
            let src = values.as_slice();
            match idx_mask {
                Mask::AllTrue(_) => {
                    for ((slot, &val), &idx) in dst.iter_mut().zip(src).zip(indices) {
                        let chunk = (offset + idx.as_()) / FL_CHUNK_SIZE;
                        slot.write(val.wrapping_add(&references[chunk]));
                    }
                }
                Mask::Values(mask_values) => {
                    let bits = mask_values.bit_buffer();
                    for (i, ((slot, &val), &idx)) in
                        dst.iter_mut().zip(src).zip(indices).enumerate()
                    {
                        if bits.value(i) {
                            let chunk = (offset + idx.as_()) / FL_CHUNK_SIZE;
                            slot.write(val.wrapping_add(&references[chunk]));
                        } else {
                            slot.write(T::zero());
                        }
                    }
                }
                Mask::AllFalse(_) => unreachable!(),
            }
            // SAFETY: every slot in `0..len` was initialized above.
            unsafe { out.set_len(len) };
            out.freeze()
        }
    };
    Ok(PrimitiveArray::new(buffer, validity))
}

fn take_blocked_narrow_t<T>(
    taken_encoded: PrimitiveArray,
    references: &[T],
    indices: &PrimitiveArray,
    idx_mask: &Mask,
    offset: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: NativePType + WrappingAdd,
    u8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    match_each_unsigned_integer_ptype!(taken_encoded.ptype(), |U| {
        take_blocked_narrow_tu::<T, U>(taken_encoded, references, indices, idx_mask, offset, ctx)
    })
}

fn take_blocked_narrow_tu<T, U>(
    taken_encoded: PrimitiveArray,
    references: &[T],
    indices: &PrimitiveArray,
    idx_mask: &Mask,
    offset: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: NativePType + WrappingAdd,
    U: NativePType + AsPrimitive<T>,
{
    match_each_integer_ptype!(indices.ptype(), |I| {
        take_blocked_narrow::<T, U, I>(
            taken_encoded,
            references,
            indices.as_slice::<I>(),
            idx_mask,
            offset,
            ctx,
        )
    })
}

fn take_blocked_narrow<T, U, I>(
    taken_encoded: PrimitiveArray,
    references: &[T],
    indices: &[I],
    idx_mask: &Mask,
    offset: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: NativePType + WrappingAdd,
    U: NativePType + AsPrimitive<T>,
    I: NativePType + AsPrimitive<usize>,
{
    let validity = taken_encoded.validity()?;
    let len = taken_encoded.len();
    if len == 0 || matches!(idx_mask, Mask::AllFalse(_)) {
        return Ok(PrimitiveArray::new(Buffer::<T>::zeroed(len), validity));
    }

    let src = taken_encoded.as_slice::<U>();
    let mut out = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
    let dst = &mut out.spare_capacity_mut()[..len];
    match idx_mask {
        Mask::AllTrue(_) => {
            for ((slot, &val), &idx) in dst.iter_mut().zip(src).zip(indices) {
                let chunk = (offset + idx.as_()) / FL_CHUNK_SIZE;
                slot.write(references[chunk].wrapping_add(&val.as_()));
            }
        }
        Mask::Values(mask_values) => {
            let bits = mask_values.bit_buffer();
            for (i, ((slot, &val), &idx)) in dst.iter_mut().zip(src).zip(indices).enumerate() {
                if bits.value(i) {
                    let chunk = (offset + idx.as_()) / FL_CHUNK_SIZE;
                    slot.write(references[chunk].wrapping_add(&val.as_()));
                } else {
                    slot.write(T::zero());
                }
            }
        }
        Mask::AllFalse(_) => unreachable!(),
    }
    // SAFETY: every slot in `0..len` was initialized above.
    unsafe { out.set_len(len) };
    Ok(PrimitiveArray::new(out.freeze(), validity))
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use crate::FoR;
    use crate::FoRArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn fa(encoded: ArrayRef, reference: Scalar) -> FoRArray {
        FoR::try_new(encoded, reference).vortex_expect("FoR array construction should succeed")
    }

    #[test]
    fn test_filter_for_array() {
        let for_array = fa(
            buffer![100i32, 101, 102, 103, 104].into_array(),
            Scalar::from(100i32),
        );
        test_filter_conformance(&for_array.into_array(), &mut SESSION.create_execution_ctx());

        let for_array = fa(
            buffer![1000u64, 1001, 1002, 1003, 1004].into_array(),
            Scalar::from(1000u64),
        );
        test_filter_conformance(&for_array.into_array(), &mut SESSION.create_execution_ctx());

        let values =
            PrimitiveArray::from_option_iter([Some(50i16), None, Some(52), Some(53), None]);
        let for_array = fa(values.into_array(), Scalar::from(50i16));
        test_filter_conformance(&for_array.into_array(), &mut SESSION.create_execution_ctx());
    }

    #[rstest]
    #[case(fa(buffer![100i32, 101, 102, 103, 104].into_array(), Scalar::from(100i32)))]
    #[case(fa(buffer![1000u64, 1001, 1002, 1003, 1004].into_array(), Scalar::from(1000u64)))]
    #[case(fa(
        PrimitiveArray::from_option_iter([Some(50i16), None, Some(52), Some(53), None]).into_array(),
        Scalar::from(50i16)
    ))]
    #[case(fa(buffer![-100i32, -99, -98, -97, -96].into_array(), Scalar::from(-100i32)))]
    #[case(fa(buffer![42i64].into_array(), Scalar::from(40i64)))]
    fn test_take_for_conformance(#[case] for_array: FoRArray) {
        test_take_conformance(&for_array.into_array(), &mut SESSION.create_execution_ctx());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::BitPackedData;
    use crate::FoR;
    use crate::FoRArray;
    use crate::for_::array::FoRArraySlotsExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn fa(encoded: ArrayRef, reference: Scalar) -> FoRArray {
        FoR::try_new(encoded, reference).vortex_expect("FoR array construction should succeed")
    }

    #[rstest]
    #[case::for_i32(fa(buffer![100i32, 101, 102, 103, 104].into_array(), Scalar::from(100i32)))]
    #[case::for_u64(fa(buffer![1000u64, 1001, 1002, 1003, 1004].into_array(), Scalar::from(1000u64)))]
    #[case::for_nullable_i16(fa(
        PrimitiveArray::from_option_iter([Some(50i16), None, Some(52), Some(53), None]).into_array(),
        Scalar::from(50i16)
    ))]
    #[case::for_nullable_i32(fa(
        PrimitiveArray::from_option_iter([Some(200i32), None, Some(202), Some(203), None]).into_array(),
        Scalar::from(200i32)
    ))]
    #[case::for_negative(fa(buffer![-100i32, -99, -98, -97, -96].into_array(), Scalar::from(-100i32)))]
    #[case::for_single(fa(buffer![42i64].into_array(), Scalar::from(40i64)))]
    #[case::for_zero_ref(fa(buffer![0u32, 1, 2, 3, 4].into_array(), Scalar::from(0u32)))]
    #[case::for_large(fa(
        PrimitiveArray::from_iter((0..1500).map(|i| 5000 + i)).into_array(),
        Scalar::from(5000i32)
    ))]
    #[case::for_very_large(fa(
        PrimitiveArray::from_iter((0..3072).map(|i| 10000 + i as i64)).into_array(),
        Scalar::from(10000i64)
    ))]
    #[case::for_large_nullable(fa(
        PrimitiveArray::from_option_iter((0..2048).map(|i| (i % 15 == 0).then_some(1000 + i))).into_array(),
        Scalar::from(1000i32)
    ))]
    #[case::for_large_deltas(fa(buffer![100i64, 200, 300, 400, 500].into_array(), Scalar::from(100i64)))]

    fn test_for_consistency(#[case] array: FoRArray) {
        test_array_consistency(&array.into_array(), &mut SESSION.create_execution_ctx());
    }

    #[rstest]
    #[case::for_i32_basic(fa(buffer![100i32, 101, 102, 103, 104].into_array(), Scalar::from(100i32)))]
    #[case::for_u32_basic(fa(buffer![1000u32, 1001, 1002, 1003, 1004].into_array(), Scalar::from(1000u32)))]
    #[case::for_i64_basic(fa(buffer![5000i64, 5001, 5002, 5003, 5004].into_array(), Scalar::from(5000i64)))]
    #[case::for_u64_basic(fa(buffer![10000u64, 10001, 10002, 10003, 10004].into_array(), Scalar::from(10000u64)))]
    #[case::for_i32_large(fa(
        PrimitiveArray::from_iter((0..100).map(|i| 2000 + i)).into_array(),
        Scalar::from(2000i32)
    ))]
    fn test_for_binary_numeric(#[case] array: FoRArray) {
        test_binary_numeric_array(&array.into_array(), &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_blocked_filter_and_take_multi_chunk_narrowed_patched_sliced() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Multi-chunk nullable i64 narrowed to u16 with patches across chunk boundaries.
        let raw = PrimitiveArray::from_option_iter((0..3500usize).map(|i| {
            if i % 23 == 0 {
                return None;
            }
            let base = -500_000i64 + (i / 1024) as i64 * 10_000_000;
            let offset = if i % 97 == 0 { 40_000 } else { (i % 50) as i64 };
            Some(base + offset)
        }));
        let for_arr = FoR::encode_chunked(raw.clone(), &mut ctx)?;
        let bp = BitPackedData::encode(for_arr.encoded(), 6, &mut ctx)?;
        let refs = for_arr.blocked_references().unwrap().clone();
        let blocked = FoR::try_new_chunked(bp.into_array(), refs, 0)?.into_array();

        // Conformance suites on multi-chunk blocked array.
        test_filter_conformance(&blocked, &mut ctx);
        test_take_conformance(&blocked, &mut ctx);

        // Sparse filter across chunk boundaries.
        let sparse_mask = Mask::from_iter((0..3500).map(|i| i % 37 == 0 || i == 1023 || i == 1024));
        assert_arrays_eq!(
            blocked.filter(sparse_mask.clone())?,
            raw.clone().into_array().filter(sparse_mask)?,
            &mut ctx
        );

        // Nullable indices take (with signed & unsigned indices, crossing chunks and duplicates).
        let take_u32 = PrimitiveArray::from_option_iter([
            Some(0u32),
            None,
            Some(1023),
            Some(1024),
            Some(3499),
            Some(1024),
            None,
            Some(2048),
        ])
        .into_array();
        assert_arrays_eq!(
            blocked.take(take_u32.clone())?,
            raw.clone().into_array().take(take_u32)?,
            &mut ctx
        );
        let take_i32 = PrimitiveArray::from_option_iter([Some(1500i32), None, Some(5), Some(3000)])
            .into_array();
        assert_arrays_eq!(
            blocked.take(take_i32.clone())?,
            raw.into_array().take(take_i32)?,
            &mut ctx
        );

        // Sliced unpatched blocked array (non-zero FoR offset) filter & take.
        let unpatched_raw = PrimitiveArray::from_iter(
            (0..3500i64).map(|i| -1_000_000 + (i / 1024) * 500_000 + (i % 100)),
        );
        let u_for = FoR::encode_chunked(unpatched_raw.clone(), &mut ctx)?;
        let u_bp = BitPackedData::encode(u_for.encoded(), 7, &mut ctx)?;
        let u_blocked = FoR::try_new_chunked(
            u_bp.into_array(),
            u_for.blocked_references().unwrap().clone(),
            0,
        )?
        .into_array();
        let sliced = u_blocked.slice(300..2600)?.optimize_ctx(&SESSION)?;
        let sliced_raw = unpatched_raw.into_array().slice(300..2600)?;

        let mask = Mask::from_iter((0..2300).map(|i| i % 13 == 0 || (700..750).contains(&i)));
        assert_arrays_eq!(
            sliced.filter(mask.clone())?,
            sliced_raw.filter(mask)?,
            &mut ctx
        );
        let idx =
            PrimitiveArray::from_option_iter([Some(0u64), Some(723), Some(724), None, Some(2299)])
                .into_array();
        assert_arrays_eq!(sliced.take(idx.clone())?, sliced_raw.take(idx)?, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_blocked_filter_gallop_patches_bw0_and_dense_bitmap() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Same-width i32 with bw=0 and many patches, so very sparse masks hit the galloping path.
        let raw_i32 = PrimitiveArray::from_iter((0..3072usize).map(|i| {
            let base = -200_000i32 + (i / 1024) as i32 * 1_000_000;
            if i % 5 == 0 { base + 500_000 } else { base }
        }));
        let for_i32 = FoR::encode_chunked(raw_i32.clone(), &mut ctx)?;
        let bp_i32 = BitPackedData::encode(for_i32.encoded(), 0, &mut ctx)?;
        let blocked_i32 = FoR::try_new_chunked(
            bp_i32.into_array(),
            for_i32.blocked_references().unwrap().clone(),
            0,
        )?
        .into_array();

        // 3 selected rows vs >600 patches triggers `p_idx.len() > true_count * 16`.
        let ultra_sparse_indices = Mask::from_indices(3072, vec![0, 1025, 3070]);
        assert_arrays_eq!(
            blocked_i32.filter(ultra_sparse_indices.clone())?,
            raw_i32.clone().into_array().filter(ultra_sparse_indices)?,
            &mut ctx
        );
        let ultra_sparse_bitmap =
            Mask::from_iter((0..3072).map(|i| i == 5 || i == 1024 || i == 2500));
        assert_arrays_eq!(
            blocked_i32.filter(ultra_sparse_bitmap.clone())?,
            raw_i32.clone().into_array().filter(ultra_sparse_bitmap)?,
            &mut ctx
        );
        // Alternating 50% bitmap exercises dense chunked SIMD compaction + linear patch ranking.
        let dense_mask = Mask::from_iter((0..3072).map(|i| (i * 7 + 3) % 2 == 0));
        assert_arrays_eq!(
            blocked_i32.filter(dense_mask.clone())?,
            raw_i32.clone().into_array().filter(dense_mask)?,
            &mut ctx
        );

        // Sliced patched blocked array (`for_offset > 0` and `patch_offset > 0`).
        let sliced = blocked_i32.slice(250..2800)?.optimize_ctx(&SESSION)?;
        let sliced_raw = raw_i32.into_array().slice(250..2800)?;
        let sliced_sparse = Mask::from_indices(2550, vec![0, 5, 774, 775, 2545]);
        assert_arrays_eq!(
            sliced.filter(sliced_sparse.clone())?,
            sliced_raw.filter(sliced_sparse)?,
            &mut ctx
        );
        let sliced_dense = Mask::from_iter((0..2550).map(|i| (i * 11 + 1) % 3 != 0));
        assert_arrays_eq!(
            sliced.filter(sliced_dense.clone())?,
            sliced_raw.filter(sliced_dense)?,
            &mut ctx
        );

        // Narrowed i64 -> u16 at bit_width=0 with patches.
        let raw_i64 = PrimitiveArray::from_iter((0..2500usize).map(|i| {
            let base = (i / 1024) as i64 * 100_000_000;
            if i % 7 == 0 { base + 20_000 } else { base }
        }));
        let for_i64 = FoR::encode_chunked(raw_i64.clone(), &mut ctx)?;
        let bp_i64 = BitPackedData::encode(for_i64.encoded(), 0, &mut ctx)?;
        let blocked_i64 = FoR::try_new_chunked(
            bp_i64.into_array(),
            for_i64.blocked_references().unwrap().clone(),
            0,
        )?
        .into_array();
        let mask_i64 = Mask::from_iter((0..2500).map(|i| i % 2 == 0));
        assert_arrays_eq!(
            blocked_i64.filter(mask_i64.clone())?,
            raw_i64.into_array().filter(mask_i64)?,
            &mut ctx
        );
        Ok(())
    }
}

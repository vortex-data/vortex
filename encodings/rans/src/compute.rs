// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-skipping `SliceReduce`, `TakeExecute`, `FilterKernel`, and fused `ZigZag` parent
//! execution for [`Rans`].

#![expect(clippy::cast_possible_truncation)]

use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_mask::MaskValues;
use vortex_zigzag::ZigZag;
use vortex_zigzag::ZigZagSlots;

use crate::Rans;
use crate::RansArrayExt;
use crate::RansData;
use crate::array::decompress_rans_mapped;
use crate::codec::BLOCK_HEADER_BYTES;
use crate::codec::BLOCK_VALUES;
use crate::codec::NUM_LANES;
use crate::codec::RansBlockHeader;
use crate::codec::decode_block_stripes;

/// Above `1/8` (`12.5%`) take selectivity, full-array decompression + primitive gather beats
/// per-block scratch decode.
const FULL_ARRAY_DECODE_RATIO: usize = 8;

/// Fused parent-execute kernel that decodes `ZigZag(Rans)` in a single pass with zero intermediate
/// unsigned buffer allocation.
#[derive(Clone, Copy, Debug)]
pub struct RansZigZagKernel;

impl ExecuteParentKernel<Rans> for RansZigZagKernel {
    type Parent = ZigZag;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Rans>,
        parent: ArrayView<'_, ZigZag>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != ZigZagSlots::ENCODED {
            return Ok(None);
        }

        let validity = match array.rans_validity() {
            Validity::Array(v) if !v.is_canonical() => {
                Validity::Array(v.execute::<BoolArray>(ctx)?.into_array())
            }
            other => other,
        };

        let decoded = match parent.dtype().as_ptype() {
            PType::I8 => decompress_rans_mapped::<i8, _>(array, validity, ctx, |u| {
                let v = u as u8;
                ((v >> 1) as i8) ^ -((v & 1) as i8)
            }),
            PType::I16 => decompress_rans_mapped::<i16, _>(array, validity, ctx, |u| {
                let v = u as u16;
                ((v >> 1) as i16) ^ -((v & 1) as i16)
            }),
            PType::I32 => decompress_rans_mapped::<i32, _>(array, validity, ctx, |u| {
                let v = u as u32;
                ((v >> 1) as i32) ^ -((v & 1) as i32)
            }),
            PType::I64 => decompress_rans_mapped::<i64, _>(array, validity, ctx, |u| {
                ((u >> 1) as i64) ^ -((u & 1) as i64)
            }),
            other => {
                vortex_bail!("ZigZag parent of Rans must have signed integer ptype, got {other}")
            }
        };
        Ok(Some(decoded.into_array()))
    }
}

impl SliceReduce for Rans {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let validity = array.rans_validity().slice(range.clone())?;
        if range.is_empty() {
            let data = RansData::try_new(
                0,
                0,
                array.has_extra_bits(),
                Arc::from(array.sym_freqs()),
                Arc::clone(array.decode_table()),
                array.block_headers().slice(0..0),
                array.payload().slice(0..0),
            )?;
            return Ok(Some(
                Rans::try_new(array.dtype().clone(), 0, data, validity)?.into_array(),
            ));
        }

        let abs_start = usize::from(array.offset()) + range.start;
        let abs_stop = usize::from(array.offset()) + range.end;
        let first_block = abs_start >> 12;
        let end_block = abs_stop.div_ceil(BLOCK_VALUES);
        let new_offset = (abs_start & (BLOCK_VALUES - 1)) as u16;

        let new_headers = array
            .block_headers()
            .slice(first_block * BLOCK_HEADER_BYTES..end_block * BLOCK_HEADER_BYTES);

        let (new_payload, new_base_offset) = if array.block_headers().is_on_host() {
            let hdr_bytes = array.block_headers().as_host();
            let first_hdr = RansBlockHeader::read_at(hdr_bytes.as_ref(), first_block);
            let last_hdr = RansBlockHeader::read_at(hdr_bytes.as_ref(), end_block - 1);
            let old_base = array.payload_base_offset() as usize;
            let start_abs_byte = first_hdr.rans_byte_offset as usize;
            let end_abs_byte = last_hdr.end_byte_offset()?;
            let rel_start = start_abs_byte - old_base;
            let rel_end = end_abs_byte - old_base;
            (
                array.payload().slice(rel_start..rel_end),
                first_hdr.rans_byte_offset,
            )
        } else {
            (array.payload().clone(), array.payload_base_offset())
        };

        let data = RansData::try_new(
            new_offset,
            new_base_offset,
            array.has_extra_bits(),
            Arc::from(array.sym_freqs()),
            Arc::clone(array.decode_table()),
            new_headers,
            new_payload,
        )?;
        Ok(Some(
            Rans::try_new(array.dtype().clone(), range.len(), data, validity)?.into_array(),
        ))
    }
}

impl TakeExecute for Rans {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if indices.is_empty() {
            let dtype = array
                .dtype()
                .union_nullability(indices.dtype().nullability());
            return Ok(Some(Canonical::empty(&dtype).into_array()));
        }
        if matches!(indices.validity()?, Validity::AllInvalid) {
            return Ok(Some(
                ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                    .into_array(),
            ));
        }
        if indices.len().saturating_mul(FULL_ARRAY_DECODE_RATIO) > array.len() {
            return Ok(None);
        }
        if u32::try_from(indices.len()).is_err() {
            return Ok(None);
        }

        let indices_prim = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let index_validity = indices_prim
            .validity()?
            .execute_mask(indices_prim.len(), ctx)?;
        if matches!(index_validity, Mask::AllFalse(_)) {
            return Ok(Some(
                ConstantArray::new(
                    Scalar::null(array.dtype().as_nullable()),
                    indices_prim.len(),
                )
                .into_array(),
            ));
        }

        let taken_validity = array
            .rans_validity()
            .take(&indices_prim.clone().into_array())?;
        let ptype = array.dtype().as_ptype();

        let result = match_each_integer_ptype!(ptype, |T| {
            match_each_integer_ptype!(indices_prim.ptype(), |I| {
                take_typed::<T, I>(
                    array,
                    indices_prim.as_slice::<I>(),
                    &index_validity,
                    taken_validity,
                    ctx,
                )?
            })
        });
        Ok(Some(result.into_array()))
    }
}

fn take_typed<T, I>(
    array: ArrayView<'_, Rans>,
    indices: &[I],
    index_validity: &Mask,
    taken_validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: NativePType,
    u64: AsPrimitive<T>,
    I: IntegerPType,
{
    let array_len = array.len();
    let offset = usize::from(array.offset());
    let num_blocks = (offset + array_len).div_ceil(BLOCK_VALUES);
    let plan = scan_take_blocks(indices, index_validity, array_len, offset, num_blocks)?;

    let out_len = indices.len();
    let mut out = BufferMut::<T>::with_capacity_in(out_len, ctx.allocator().clone());
    // SAFETY: Every slot in `0..out_len` is initialized below (including null index slots).
    unsafe { out.set_len(out_len) };

    if plan.block_ordered {
        take_ordered_blocks::<T, I>(array, indices, index_validity, &plan, out.as_mut_slice());
    } else {
        take_scattered_blocks::<T, I>(array, indices, index_validity, &plan, out.as_mut_slice());
    }

    Ok(PrimitiveArray::new(out.freeze(), taken_validity))
}

struct TakeBlockPlan {
    hits_per_block: Vec<u32>,
    max_stripe_per_block: Vec<u8>,
    block_ordered: bool,
}

fn scan_take_blocks<I: IntegerPType>(
    indices: &[I],
    index_validity: &Mask,
    array_len: usize,
    offset: usize,
    num_blocks: usize,
) -> VortexResult<TakeBlockPlan> {
    let mut hits_per_block = vec![0u32; num_blocks];
    let mut max_stripe_per_block = vec![0u8; num_blocks];
    let mut block_ordered = true;
    let mut prev_block = 0usize;

    match index_validity {
        Mask::AllTrue(_) => {
            for &raw_idx in indices {
                let idx = checked_index(raw_idx, array_len)?;
                let pos = offset + idx;
                let b = pos >> 12;
                let stripe = ((pos & (BLOCK_VALUES - 1)) >> 5) as u8;
                if b < prev_block {
                    block_ordered = false;
                }
                prev_block = b;
                hits_per_block[b] += 1;
                max_stripe_per_block[b] = max_stripe_per_block[b].max(stripe);
            }
        }
        Mask::Values(mv) => {
            let bits = mv.bit_buffer();
            for (out_i, &raw_idx) in indices.iter().enumerate() {
                if bits.value(out_i) {
                    let idx = checked_index(raw_idx, array_len)?;
                    let pos = offset + idx;
                    let b = pos >> 12;
                    let stripe = ((pos & (BLOCK_VALUES - 1)) >> 5) as u8;
                    if b < prev_block {
                        block_ordered = false;
                    }
                    prev_block = b;
                    hits_per_block[b] += 1;
                    max_stripe_per_block[b] = max_stripe_per_block[b].max(stripe);
                }
            }
        }
        Mask::AllFalse(_) => unreachable!("handled by caller"),
    }

    Ok(TakeBlockPlan {
        hits_per_block,
        max_stripe_per_block,
        block_ordered,
    })
}

fn take_ordered_blocks<T, I>(
    array: ArrayView<'_, Rans>,
    indices: &[I],
    index_validity: &Mask,
    plan: &TakeBlockPlan,
    out_slice: &mut [T],
) where
    T: NativePType,
    u64: AsPrimitive<T>,
    I: IntegerPType,
{
    let offset = usize::from(array.offset());
    let headers_bytes = array.block_headers().as_host();
    let payload_bytes = array.payload().as_host();
    let table = array.decode_table();
    let base_offset = array.payload_base_offset();
    let has_extra_bits = array.has_extra_bits();
    let mut scratch: [MaybeUninit<T>; BLOCK_VALUES] =
        [const { MaybeUninit::uninit() }; BLOCK_VALUES];
    let mut decoded_block = usize::MAX;

    let mut read_at = |raw_idx: I| -> T {
        let pos = offset + AsPrimitive::<usize>::as_(raw_idx);
        let b = pos >> 12;
        let in_block = pos & (BLOCK_VALUES - 1);
        if b != decoded_block {
            let n_stripes = usize::from(plan.max_stripe_per_block[b]) + 1;
            decode_block_stripes::<T>(
                table,
                headers_bytes.as_ref(),
                payload_bytes.as_ref(),
                base_offset,
                b,
                n_stripes,
                has_extra_bits,
                &mut scratch[..n_stripes * NUM_LANES],
            );
            decoded_block = b;
        }
        // SAFETY: `in_block <= max_stripe_per_block[b] * 32 + 31 < n_stripes * 32`, which was
        // initialized by `decode_block_stripes`.
        unsafe { scratch[in_block].assume_init() }
    };

    match index_validity {
        Mask::AllTrue(_) => {
            for (dst, &raw_idx) in out_slice.iter_mut().zip(indices) {
                *dst = read_at(raw_idx);
            }
        }
        Mask::Values(mv) => {
            let bits = mv.bit_buffer();
            for (out_i, (dst, &raw_idx)) in out_slice.iter_mut().zip(indices).enumerate() {
                *dst = if bits.value(out_i) {
                    read_at(raw_idx)
                } else {
                    T::default()
                };
            }
        }
        Mask::AllFalse(_) => unreachable!(),
    }
}

fn take_scattered_blocks<T, I>(
    array: ArrayView<'_, Rans>,
    indices: &[I],
    index_validity: &Mask,
    plan: &TakeBlockPlan,
    out_slice: &mut [T],
) where
    T: NativePType,
    u64: AsPrimitive<T>,
    I: IntegerPType,
{
    let offset = usize::from(array.offset());
    let num_blocks = plan.hits_per_block.len();
    let mut offsets = vec![0u32; num_blocks + 1];
    for b in 0..num_blocks {
        offsets[b + 1] = offsets[b] + plan.hits_per_block[b];
    }
    let total_valid_hits = offsets[num_blocks] as usize;
    let mut cursor = offsets[..num_blocks].to_vec();
    let mut grouped = vec![(0u32, 0u16); total_valid_hits];

    match index_validity {
        Mask::AllTrue(_) => {
            for (out_i, &raw_idx) in indices.iter().enumerate() {
                let pos = offset + AsPrimitive::<usize>::as_(raw_idx);
                let b = pos >> 12;
                let slot = cursor[b] as usize;
                cursor[b] += 1;
                grouped[slot] = (out_i as u32, (pos & (BLOCK_VALUES - 1)) as u16);
            }
        }
        Mask::Values(mv) => {
            let bits = mv.bit_buffer();
            for (out_i, &raw_idx) in indices.iter().enumerate() {
                if bits.value(out_i) {
                    let pos = offset + AsPrimitive::<usize>::as_(raw_idx);
                    let b = pos >> 12;
                    let slot = cursor[b] as usize;
                    cursor[b] += 1;
                    grouped[slot] = (out_i as u32, (pos & (BLOCK_VALUES - 1)) as u16);
                } else {
                    out_slice[out_i] = T::default();
                }
            }
        }
        Mask::AllFalse(_) => unreachable!(),
    }

    let headers_bytes = array.block_headers().as_host();
    let payload_bytes = array.payload().as_host();
    let table = array.decode_table();
    let base_offset = array.payload_base_offset();
    let has_extra_bits = array.has_extra_bits();
    let mut scratch: [MaybeUninit<T>; BLOCK_VALUES] =
        [const { MaybeUninit::uninit() }; BLOCK_VALUES];

    for b in 0..num_blocks {
        let start_hit = offsets[b] as usize;
        let end_hit = offsets[b + 1] as usize;
        if start_hit == end_hit {
            continue;
        }
        let n_stripes = usize::from(plan.max_stripe_per_block[b]) + 1;
        decode_block_stripes::<T>(
            table,
            headers_bytes.as_ref(),
            payload_bytes.as_ref(),
            base_offset,
            b,
            n_stripes,
            has_extra_bits,
            &mut scratch[..n_stripes * NUM_LANES],
        );
        for &(out_i, in_block) in &grouped[start_hit..end_hit] {
            // SAFETY: `in_block < n_stripes * 32`, initialized above by `decode_block_stripes`.
            out_slice[out_i as usize] = unsafe { scratch[usize::from(in_block)].assume_init() };
        }
    }
}

#[inline]
fn checked_index<I: IntegerPType>(raw_idx: I, array_len: usize) -> VortexResult<usize> {
    let Some(idx) = raw_idx.to_usize() else {
        vortex_bail!("Take index {raw_idx} cannot convert to usize");
    };
    if idx >= array_len {
        vortex_bail!("Take index {idx} out of bounds for RansArray of length {array_len}");
    }
    Ok(idx)
}

impl FilterKernel for Rans {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask_values = match mask {
            Mask::AllTrue(_) => return Ok(Some(array.array().clone())),
            Mask::AllFalse(_) => return Ok(Some(Canonical::empty(array.dtype()).into_array())),
            Mask::Values(values) => values,
        };

        let validity = array.rans_validity().filter(mask)?;
        let ptype = array.dtype().as_ptype();
        let filtered = match_each_integer_ptype!(ptype, |T| {
            filter_typed::<T>(array, mask_values, validity, ctx)
        });
        Ok(Some(filtered.into_array()))
    }
}

fn filter_typed<T>(
    array: ArrayView<'_, Rans>,
    mask_values: &MaskValues,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> PrimitiveArray
where
    T: NativePType,
    u64: AsPrimitive<T>,
{
    let true_count = mask_values.true_count();
    let mut out = BufferMut::<T>::with_capacity_in(true_count, ctx.allocator().clone());
    // SAFETY: Exactly `true_count` elements are written into `out_slice` below.
    unsafe { out.set_len(true_count) };
    let out_slice = out.as_mut_slice();

    let offset = usize::from(array.offset());
    let headers_bytes = array.block_headers().as_host();
    let payload_bytes = array.payload().as_host();
    let table = array.decode_table();
    let base_offset = array.payload_base_offset();
    let has_extra_bits = array.has_extra_bits();

    let mut scratch: [MaybeUninit<T>; BLOCK_VALUES] =
        [const { MaybeUninit::uninit() }; BLOCK_VALUES];
    let mut written = 0usize;

    if let Some(indices) = mask_values.cached_indices() {
        let mut rest = indices;
        while let Some(&first_idx) = rest.first() {
            let block_idx = (first_idx + offset) >> 12;
            let block_abs_start = block_idx * BLOCK_VALUES;
            let block_logical_end = block_abs_start + BLOCK_VALUES - offset;
            let count = rest.partition_point(|&i| i < block_logical_end);
            let block_indices = &rest[..count];
            let last_in_block = block_indices[count - 1] + offset - block_abs_start;
            let n_stripes = (last_in_block >> 5) + 1;

            decode_block_stripes::<T>(
                table,
                headers_bytes.as_ref(),
                payload_bytes.as_ref(),
                base_offset,
                block_idx,
                n_stripes,
                has_extra_bits,
                &mut scratch[..n_stripes * NUM_LANES],
            );

            for (dst, &orig_idx) in out_slice[written..written + count]
                .iter_mut()
                .zip(block_indices)
            {
                let in_block = orig_idx + offset - block_abs_start;
                // SAFETY: `in_block <= last_in_block < n_stripes * 32`, initialized above.
                *dst = unsafe { scratch[in_block].assume_init() };
            }
            written += count;
            rest = &rest[count..];
        }
    } else {
        let bits = mask_values.bit_buffer();
        let len = bits.len();
        let num_blocks = (offset + len).div_ceil(BLOCK_VALUES);
        let mut positions = [0u16; BLOCK_VALUES];

        for block_idx in 0..num_blocks {
            let block_abs_start = block_idx * BLOCK_VALUES;
            let start = block_abs_start.saturating_sub(offset);
            let end = (block_abs_start + BLOCK_VALUES - offset).min(len);
            let count = bits.count_range(start, end);
            if count == 0 {
                continue;
            }

            let in_block_base = start + offset - block_abs_start;
            let mut pos_idx = 0usize;
            bits.slice(start..end).for_each_set_index(|rel| {
                positions[pos_idx] = (in_block_base + rel) as u16;
                pos_idx += 1;
            });
            debug_assert_eq!(pos_idx, count);

            let last_in_block = usize::from(positions[count - 1]);
            let n_stripes = (last_in_block >> 5) + 1;

            decode_block_stripes::<T>(
                table,
                headers_bytes.as_ref(),
                payload_bytes.as_ref(),
                base_offset,
                block_idx,
                n_stripes,
                has_extra_bits,
                &mut scratch[..n_stripes * NUM_LANES],
            );

            for (dst, &in_block) in out_slice[written..written + count]
                .iter_mut()
                .zip(&positions[..count])
            {
                // SAFETY: `in_block <= last_in_block < n_stripes * 32`, initialized above.
                *dst = unsafe { scratch[usize::from(in_block)].assume_init() };
            }
            written += count;
        }
    }

    debug_assert_eq!(written, true_count);
    PrimitiveArray::new(out.freeze(), validity)
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-skipping `SliceReduce`, `TakeExecute`, `FilterKernel`, fused `ZigZag` parent execution,
//! and direct spine-walking `CompareKernel` predicate pushdown for [`Pivco`].

#![expect(clippy::cast_possible_truncation)]

use std::mem::MaybeUninit;
use std::ops::Range;

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
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::compare::CompareKernel;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityVTable;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_mask::MaskValues;
use vortex_zigzag::ZigZag;
use vortex_zigzag::ZigZagSlots;

use crate::array::ConstantCompareResult;
use crate::array::Pivco;
use crate::array::PivcoData;
use crate::array::num_blocks_for;

/// Above `1/8` (`12.5%`) take/filter selectivity, full-array decompression + primitive gather
/// beats per-block scratch decode.
const FULL_ARRAY_DECODE_RATIO: usize = 8;

/// At or below `4` hits in a block, top-down `O(depth)` `decode_value_at` beats rebuilding the
/// full block wavelet tree.
const SCALAR_BLOCK_HIT_THRESHOLD: usize = 4;

/// Fused parent-execute kernel that decodes `ZigZag(Pivco)` in a single pass with zero
/// intermediate unsigned buffer allocation.
#[derive(Clone, Copy, Debug)]
pub struct PivcoZigZagKernel;

impl ExecuteParentKernel<Pivco> for PivcoZigZagKernel {
    type Parent = ZigZag;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Pivco>,
        parent: ArrayView<'_, ZigZag>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != ZigZagSlots::ENCODED || !array.data().is_on_host() {
            return Ok(None);
        }

        let validity = match Pivco::validity(array)? {
            Validity::Array(v) if !v.is_canonical() => {
                Validity::Array(v.execute::<BoolArray>(ctx)?.into_array())
            }
            other => other,
        };

        let data = array.data();
        let len = array.len();
        let decoded = match parent.dtype().as_ptype() {
            PType::I8 => data.decompress_mapped::<i8, _>(len, validity, ctx, |u| {
                let v = u as u8;
                ((v >> 1) as i8) ^ -((v & 1) as i8)
            })?,
            PType::I16 => data.decompress_mapped::<i16, _>(len, validity, ctx, |u| {
                let v = u as u16;
                ((v >> 1) as i16) ^ -((v & 1) as i16)
            })?,
            PType::I32 => data.decompress_mapped::<i32, _>(len, validity, ctx, |u| {
                let v = u as u32;
                ((v >> 1) as i32) ^ -((v & 1) as i32)
            })?,
            PType::I64 => data.decompress_mapped::<i64, _>(len, validity, ctx, |u| {
                ((u >> 1) as i64) ^ -((u & 1) as i64)
            })?,
            other => {
                vortex_bail!("ZigZag parent of Pivco must have signed integer ptype, got {other}")
            }
        };
        Ok(Some(decoded.into_array()))
    }
}

impl SliceReduce for Pivco {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let validity = Self::validity(array)?.slice(range.clone())?;
        let data = array.data().slice(array.len(), range.clone())?;
        Ok(Some(
            Self::try_new(array.dtype().clone(), range.len(), data, validity)?.into_array(),
        ))
    }
}

impl TakeExecute for Pivco {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        take_data(
            array.data(),
            array.dtype(),
            array.len(),
            Self::validity(array)?,
            indices,
            ctx,
        )
    }
}

fn take_data(
    data: &PivcoData,
    dtype: &DType,
    array_len: usize,
    validity: Validity,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if !data.is_on_host() {
        return Ok(None);
    }
    if indices.is_empty() {
        let out_dtype = dtype.union_nullability(indices.dtype().nullability());
        return Ok(Some(Canonical::empty(&out_dtype).into_array()));
    }
    if matches!(indices.validity()?, Validity::AllInvalid) {
        return Ok(Some(
            ConstantArray::new(Scalar::null(dtype.as_nullable()), indices.len()).into_array(),
        ));
    }
    if indices.len().saturating_mul(FULL_ARRAY_DECODE_RATIO) > array_len {
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
            ConstantArray::new(Scalar::null(dtype.as_nullable()), indices_prim.len()).into_array(),
        ));
    }

    let taken_validity = validity.take(&indices_prim.clone().into_array())?;
    let ptype = dtype.as_ptype();

    let result = match_each_integer_ptype!(ptype, |T| {
        match_each_integer_ptype!(indices_prim.ptype(), |I| {
            take_typed::<T, I>(
                data,
                array_len,
                indices_prim.as_slice::<I>(),
                &index_validity,
                taken_validity,
                ctx,
            )?
        })
    });
    Ok(Some(result.into_array()))
}

fn take_typed<T, I>(
    data: &PivcoData,
    array_len: usize,
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
    let block_shift = data.block_shift();
    let block_values = 1usize << block_shift;
    let offset = usize::from(data.offset());
    let num_blocks = (offset + array_len).div_ceil(block_values);
    let plan = scan_take_blocks(
        indices,
        index_validity,
        array_len,
        offset,
        block_shift,
        num_blocks,
    )?;

    let out_len = indices.len();
    let mut out = BufferMut::<T>::with_capacity_in(out_len, ctx.allocator().clone());
    let out_uninit = &mut out.spare_capacity_mut()[..out_len];

    if plan.block_ordered {
        take_ordered_blocks::<T, I>(data, array_len, indices, index_validity, &plan, out_uninit)?;
    } else {
        take_scattered_blocks::<T, I>(data, array_len, indices, index_validity, &plan, out_uninit)?;
    }

    // SAFETY: Every slot in `0..out_len` was initialized via `MaybeUninit::write` above.
    unsafe { out.set_len(out_len) };
    Ok(PrimitiveArray::new(out.freeze(), taken_validity))
}

struct TakeBlockPlan {
    hits_per_block: Vec<u32>,
    max_pos_per_block: Vec<u16>,
    block_ordered: bool,
}

fn scan_take_blocks<I: IntegerPType>(
    indices: &[I],
    index_validity: &Mask,
    array_len: usize,
    offset: usize,
    block_shift: u8,
    num_blocks: usize,
) -> VortexResult<TakeBlockPlan> {
    let block_mask = (1usize << block_shift) - 1;
    let mut hits_per_block = vec![0u32; num_blocks];
    let mut max_pos_per_block = vec![0u16; num_blocks];
    let mut block_ordered = true;
    let mut prev_block = 0usize;

    match index_validity {
        Mask::AllTrue(_) => {
            for &raw_idx in indices {
                let idx = checked_index(raw_idx, array_len)?;
                let pos = offset + idx;
                let b = pos >> block_shift;
                let in_b = (pos & block_mask) as u16;
                if b < prev_block {
                    block_ordered = false;
                }
                prev_block = b;
                hits_per_block[b] += 1;
                max_pos_per_block[b] = max_pos_per_block[b].max(in_b);
            }
        }
        Mask::Values(mv) => {
            for (start, end) in mv.bit_buffer().set_slices() {
                for &raw_idx in &indices[start..end] {
                    let idx = checked_index(raw_idx, array_len)?;
                    let pos = offset + idx;
                    let b = pos >> block_shift;
                    let in_b = (pos & block_mask) as u16;
                    if b < prev_block {
                        block_ordered = false;
                    }
                    prev_block = b;
                    hits_per_block[b] += 1;
                    max_pos_per_block[b] = max_pos_per_block[b].max(in_b);
                }
            }
        }
        Mask::AllFalse(_) => unreachable!("handled by caller"),
    }

    Ok(TakeBlockPlan {
        hits_per_block,
        max_pos_per_block,
        block_ordered,
    })
}

fn take_ordered_blocks<T, I>(
    data: &PivcoData,
    array_len: usize,
    indices: &[I],
    index_validity: &Mask,
    plan: &TakeBlockPlan,
    out_uninit: &mut [MaybeUninit<T>],
) -> VortexResult<()>
where
    T: NativePType,
    u64: AsPrimitive<T>,
    I: IntegerPType,
{
    let block_shift = data.block_shift();
    let block_values = 1usize << block_shift;
    let block_mask = block_values - 1;
    let offset = usize::from(data.offset());

    let mut decoder = data.block_decoder(array_len)?;
    let mut scratch: Option<Vec<MaybeUninit<T>>> = None;
    let mut cur_block = usize::MAX;

    let mut write_valid = |dst: &mut MaybeUninit<T>, raw_idx: I| -> VortexResult<()> {
        let pos = offset + raw_idx.as_();
        let b = pos >> block_shift;
        let in_block = pos & block_mask;
        if (plan.hits_per_block[b] as usize) <= SCALAR_BLOCK_HIT_THRESHOLD {
            let val: T = decoder.decode_value_at(b, in_block)?.as_();
            dst.write(val);
        } else {
            let buf = scratch.get_or_insert_with(|| vec![MaybeUninit::uninit(); block_values]);
            if b != cur_block {
                let end = usize::from(plan.max_pos_per_block[b]) + 1;
                decoder.decode_range_into::<T, _>(b, 0, end, &mut buf[..end], |u| u.as_())?;
                cur_block = b;
            }
            // SAFETY: `in_block <= max_pos_per_block[b]`, initialized above.
            dst.write(unsafe { buf[in_block].assume_init() });
        }
        Ok(())
    };

    match index_validity {
        Mask::AllTrue(_) => {
            for (dst, &raw_idx) in out_uninit.iter_mut().zip(indices) {
                write_valid(dst, raw_idx)?;
            }
        }
        Mask::Values(mv) => {
            let mut cursor = 0usize;
            for (start, end) in mv.bit_buffer().set_slices() {
                for dst in &mut out_uninit[cursor..start] {
                    dst.write(T::default());
                }
                for (dst, &raw_idx) in out_uninit[start..end].iter_mut().zip(&indices[start..end]) {
                    write_valid(dst, raw_idx)?;
                }
                cursor = end;
            }
            for dst in &mut out_uninit[cursor..] {
                dst.write(T::default());
            }
        }
        Mask::AllFalse(_) => unreachable!("handled by caller"),
    }
    Ok(())
}

fn take_scattered_blocks<T, I>(
    data: &PivcoData,
    array_len: usize,
    indices: &[I],
    index_validity: &Mask,
    plan: &TakeBlockPlan,
    out_uninit: &mut [MaybeUninit<T>],
) -> VortexResult<()>
where
    T: NativePType,
    u64: AsPrimitive<T>,
    I: IntegerPType,
{
    let block_shift = data.block_shift();
    let block_values = 1usize << block_shift;
    let block_mask = block_values - 1;
    let num_blocks = plan.hits_per_block.len();
    let mut block_starts = vec![0u32; num_blocks + 1];
    let mut total_valid = 0u32;
    for (b, &hits) in plan.hits_per_block.iter().enumerate() {
        block_starts[b] = total_valid;
        total_valid += hits;
    }
    block_starts[num_blocks] = total_valid;

    let mut write_cursors = block_starts[..num_blocks].to_vec();
    let mut grouped_out_idx = vec![0u32; total_valid as usize];
    let mut grouped_in_block = vec![0u16; total_valid as usize];
    let offset = usize::from(data.offset());

    match index_validity {
        Mask::AllTrue(_) => {
            for (out_i, &raw_idx) in indices.iter().enumerate() {
                let pos = offset + raw_idx.as_();
                let b = pos >> block_shift;
                let slot = write_cursors[b] as usize;
                write_cursors[b] += 1;
                grouped_out_idx[slot] = out_i as u32;
                grouped_in_block[slot] = (pos & block_mask) as u16;
            }
        }
        Mask::Values(mv) => {
            let mut cursor = 0usize;
            for (start, end) in mv.bit_buffer().set_slices() {
                for dst in &mut out_uninit[cursor..start] {
                    dst.write(T::default());
                }
                for (out_i, &raw_idx) in (start..end).zip(&indices[start..end]) {
                    let pos = offset + raw_idx.as_();
                    let b = pos >> block_shift;
                    let slot = write_cursors[b] as usize;
                    write_cursors[b] += 1;
                    grouped_out_idx[slot] = out_i as u32;
                    grouped_in_block[slot] = (pos & block_mask) as u16;
                }
                cursor = end;
            }
            for dst in &mut out_uninit[cursor..] {
                dst.write(T::default());
            }
        }
        Mask::AllFalse(_) => unreachable!("handled by caller"),
    }

    let mut decoder = data.block_decoder(array_len)?;
    let mut scratch: Option<Vec<MaybeUninit<T>>> = None;

    for b in 0..num_blocks {
        let start = block_starts[b] as usize;
        let end = block_starts[b + 1] as usize;
        let count = end - start;
        if count == 0 {
            continue;
        }
        if count <= SCALAR_BLOCK_HIT_THRESHOLD {
            for slot in start..end {
                let out_i = grouped_out_idx[slot] as usize;
                let in_block = usize::from(grouped_in_block[slot]);
                let val: T = decoder.decode_value_at(b, in_block)?.as_();
                out_uninit[out_i].write(val);
            }
        } else {
            let buf = scratch.get_or_insert_with(|| vec![MaybeUninit::uninit(); block_values]);
            let decode_end = usize::from(plan.max_pos_per_block[b]) + 1;
            decoder
                .decode_range_into::<T, _>(b, 0, decode_end, &mut buf[..decode_end], |u| u.as_())?;
            for slot in start..end {
                let out_i = grouped_out_idx[slot] as usize;
                let in_block = usize::from(grouped_in_block[slot]);
                // SAFETY: `in_block <= max_pos_per_block[b]`, initialized above.
                out_uninit[out_i].write(unsafe { buf[in_block].assume_init() });
            }
        }
    }
    Ok(())
}

#[inline]
fn checked_index<I: IntegerPType>(raw_idx: I, array_len: usize) -> VortexResult<usize> {
    if let Some(idx) = raw_idx.to_usize()
        && idx < array_len
    {
        Ok(idx)
    } else {
        vortex_bail!("Take index {raw_idx} out of bounds for PivcoArray of length {array_len}")
    }
}

impl FilterKernel for Pivco {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if matches!(mask, Mask::AllTrue(_)) {
            return Ok(Some(array.array().clone()));
        }
        filter_data(
            array.data(),
            array.dtype(),
            array.len(),
            Self::validity(array)?,
            mask,
            ctx,
        )
    }
}

fn filter_data(
    data: &PivcoData,
    dtype: &DType,
    array_len: usize,
    validity: Validity,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if !data.is_on_host() {
        return Ok(None);
    }
    let mask_values = match mask {
        Mask::AllTrue(_) => unreachable!("handled by caller"),
        Mask::AllFalse(_) => return Ok(Some(Canonical::empty(dtype).into_array())),
        Mask::Values(values) => values,
    };

    // High-selectivity filters (> 1/8) are faster via canonical decode + primitive filter.
    if mask_values
        .true_count()
        .saturating_mul(FULL_ARRAY_DECODE_RATIO)
        > array_len
    {
        return Ok(None);
    }

    let filtered_validity = validity.filter(mask)?;
    let ptype = dtype.as_ptype();
    let filtered = match_each_integer_ptype!(ptype, |T| {
        filter_typed::<T>(data, array_len, mask_values, filtered_validity, ctx)?
    });
    Ok(Some(filtered.into_array()))
}

fn filter_typed<T>(
    data: &PivcoData,
    array_len: usize,
    mask_values: &MaskValues,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: NativePType,
    u64: AsPrimitive<T>,
{
    let true_count = mask_values.true_count();
    let mut out = BufferMut::<T>::with_capacity_in(true_count, ctx.allocator().clone());
    let out_uninit = &mut out.spare_capacity_mut()[..true_count];

    let block_shift = data.block_shift();
    let block_values = 1usize << block_shift;
    let offset = usize::from(data.offset());
    let num_blocks = num_blocks_for(block_shift, data.offset(), array_len);

    let mut decoder = data.block_decoder(array_len)?;
    let mut scratch: Option<Vec<MaybeUninit<T>>> = None;
    let mut written = 0usize;

    if let Some(indices) = mask_values.cached_indices() {
        let mut rest = indices;
        while let Some(&first_idx) = rest.first() {
            let block_idx = (first_idx + offset) >> block_shift;
            let block_abs_start = block_idx * block_values;
            let block_logical_end = block_abs_start + block_values - offset;
            let count = rest.partition_point(|&i| i < block_logical_end);
            let block_indices = &rest[..count];

            if count <= SCALAR_BLOCK_HIT_THRESHOLD {
                for (dst, &orig_idx) in out_uninit[written..written + count]
                    .iter_mut()
                    .zip(block_indices)
                {
                    let in_block = orig_idx + offset - block_abs_start;
                    let val: T = decoder.decode_value_at(block_idx, in_block)?.as_();
                    dst.write(val);
                }
            } else {
                let buf = scratch.get_or_insert_with(|| vec![MaybeUninit::uninit(); block_values]);
                let first_in_block = block_indices[0] + offset - block_abs_start;
                let last_in_block = block_indices[count - 1] + offset - block_abs_start;
                let decode_end = last_in_block + 1;

                decoder.decode_range_into::<T, _>(
                    block_idx,
                    first_in_block,
                    decode_end,
                    &mut buf[first_in_block..decode_end],
                    |u| u.as_(),
                )?;

                for (dst, &orig_idx) in out_uninit[written..written + count]
                    .iter_mut()
                    .zip(block_indices)
                {
                    let in_block = orig_idx + offset - block_abs_start;
                    // SAFETY: `first_in_block <= in_block < decode_end`, initialized above.
                    dst.write(unsafe { buf[in_block].assume_init() });
                }
            }
            written += count;
            rest = &rest[count..];
        }
    } else {
        let bits = mask_values.bit_buffer();
        let len = bits.len();
        let mut positions: Vec<u16> = Vec::new();

        for block_idx in 0..num_blocks {
            let block_abs_start = block_idx * block_values;
            let start = block_abs_start.saturating_sub(offset);
            let end = (block_abs_start + block_values - offset).min(len);
            let count = bits.count_range(start, end);
            if count == 0 {
                continue;
            }

            let in_block_base = start + offset - block_abs_start;
            positions.clear();
            bits.slice(start..end).for_each_set_index(|rel| {
                positions.push((in_block_base + rel) as u16);
            });
            debug_assert_eq!(positions.len(), count);

            if count <= SCALAR_BLOCK_HIT_THRESHOLD {
                for (dst, &in_block) in out_uninit[written..written + count]
                    .iter_mut()
                    .zip(&positions)
                {
                    let val: T = decoder
                        .decode_value_at(block_idx, usize::from(in_block))?
                        .as_();
                    dst.write(val);
                }
            } else {
                let buf = scratch.get_or_insert_with(|| vec![MaybeUninit::uninit(); block_values]);
                let first_in_block = usize::from(positions[0]);
                let decode_end = usize::from(positions[count - 1]) + 1;
                decoder.decode_range_into::<T, _>(
                    block_idx,
                    first_in_block,
                    decode_end,
                    &mut buf[first_in_block..decode_end],
                    |u| u.as_(),
                )?;

                for (dst, &in_block) in out_uninit[written..written + count]
                    .iter_mut()
                    .zip(&positions)
                {
                    // SAFETY: `first_in_block <= in_block < decode_end`, initialized above.
                    dst.write(unsafe { buf[usize::from(in_block)].assume_init() });
                }
            }
            written += count;
        }
    }

    debug_assert_eq!(written, true_count);
    // SAFETY: Exactly `true_count` elements were written via `MaybeUninit::write` above.
    unsafe { out.set_len(true_count) };
    Ok(PrimitiveArray::new(out.freeze(), validity))
}

// ============================================================================
// Direct Spine-Walking `CompareKernel` Pushdown (`==`, `!=`, `<`, `<=`, `>`, `>=`)
// ============================================================================

impl CompareKernel for Pivco {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        compare_data(
            lhs.data(),
            lhs.dtype(),
            lhs.len(),
            Self::validity(lhs)?,
            rhs,
            operator,
        )
    }
}

fn compare_data(
    data: &PivcoData,
    lhs_dtype: &DType,
    len: usize,
    lhs_validity: Validity,
    rhs: &ArrayRef,
    operator: CompareOperator,
) -> VortexResult<Option<ArrayRef>> {
    if !data.is_on_host() {
        return Ok(None);
    }
    let Some(const_scalar) = rhs.as_constant() else {
        return Ok(None);
    };
    let Some(pscalar) = const_scalar.as_primitive_opt() else {
        return Ok(None);
    };
    if pscalar.ptype() != lhs_dtype.as_ptype() {
        return Ok(None);
    }

    let pvalue = pscalar
        .pvalue()
        .vortex_expect("compare adaptor handles null constants");
    let nullability = lhs_dtype.nullability() | rhs.dtype().nullability();

    // Extract `rhs` as a non-negative `u64` or handle negative signed constants in `O(1)`.
    // Validated `PivcoArray` trees guarantee every element is non-negative and fits in `lhs_dtype`.
    let rhs_u64: u64 = if pvalue.ptype().is_signed_int() {
        let signed = pvalue
            .cast::<i64>()
            .vortex_expect("signed integer fits i64");
        if signed < 0 {
            let const_ans = match operator {
                CompareOperator::Eq | CompareOperator::Lt | CompareOperator::Lte => false,
                CompareOperator::NotEq | CompareOperator::Gt | CompareOperator::Gte => true,
            };
            return constant_bool_preserving_validity(len, lhs_validity, const_ans, nullability)
                .map(Some);
        }
        signed as u64
    } else {
        pvalue
            .cast::<u64>()
            .vortex_expect("unsigned integer fits u64")
    };

    match data.compare_constant_u64(len, rhs_u64, operator)? {
        Some(ConstantCompareResult::Constant(val)) => {
            constant_bool_preserving_validity(len, lhs_validity, val, nullability).map(Some)
        }
        Some(ConstantCompareResult::Bits(bits)) => {
            let validity = lhs_validity.union_nullability(nullability);
            Ok(Some(BoolArray::new(bits, validity).into_array()))
        }
        None => Ok(None),
    }
}

fn constant_bool_preserving_validity(
    len: usize,
    lhs_validity: Validity,
    value: bool,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    Ok(match lhs_validity.union_nullability(nullability) {
        Validity::NonNullable | Validity::AllValid => {
            ConstantArray::new(Scalar::bool(value, nullability), len).into_array()
        }
        Validity::AllInvalid => {
            ConstantArray::new(Scalar::null(DType::Bool(nullability)), len).into_array()
        }
        validity @ Validity::Array(_) => {
            BoolArray::new(BitBuffer::full(value, len), validity).into_array()
        }
    })
}

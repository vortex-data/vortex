// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chunk-pruned `filter` kernel for [`Affine`].

use fastlanes::BitPacking;
use fastlanes::FoR;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingAdd;
use num_traits::WrappingMul;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_buffer::filter::filter_chunked;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use super::AffineNarrowBitPackedChunks;
use super::AffineSameWidthBitPackedChunks;
use super::resolve_aligned_bitpacked;
use crate::Affine;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::affine::array::AffineArrayExt;
use crate::affine::array::AffineArraySlotsExt;
use crate::affine::array::affine_decompress::Params;
use crate::affine::array::slope_term_split;
use crate::affine::array::split_slope;

impl FilterKernel for Affine {
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

        if let Some(bp_arr) = resolve_aligned_bitpacked(array, ctx)? {
            let bp = bp_arr.as_::<BitPacked>();
            let bit_width = bp.bit_width();
            return match_each_integer_ptype!(array.ptype(), |T| {
                filter_bitpacked_typed::<T>(array, bp, bit_width, mask, mask_values, ctx).map(Some)
            });
        }

        let filtered_encoded = array
            .encoded()
            .filter(mask.clone())?
            .execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(array.ptype(), |T| {
            filter_residuals_typed::<T>(array, filtered_encoded, mask_values, ctx).map(Some)
        })
    }
}

fn filter_bitpacked_typed<T>(
    array: ArrayView<'_, Affine>,
    bp: ArrayView<'_, BitPacked>,
    bit_width: u8,
    mask: &Mask,
    mask_values: &MaskValues,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: PhysicalPType<Physical: BitPacking + FoR + AsPrimitive<u64>>
        + AsPrimitive<T::Physical>
        + AsPrimitive<u64>
        + PrimInt
        + WrappingAdd
        + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    let params = Params::<T>::execute(array, ctx)?;
    let validity = array.validity()?.filter(mask)?;
    let encoded_ptype = bp.dtype().as_ptype();
    let bit_width = usize::from(bit_width);

    let values = if encoded_ptype.byte_width() < T::PTYPE.byte_width() {
        match_each_unsigned_integer_ptype!(encoded_ptype, |U| {
            filter_bitpacked_narrow::<T, U>(bp, bit_width, &params, mask_values, ctx.allocator())
        })
    } else {
        let chunks = AffineSameWidthBitPackedChunks::<T>::new(
            bp.packed_slice::<T::Physical>(),
            bit_width,
            &params,
        );
        filter_chunked(
            &chunks,
            params.offset,
            mask_values.selection(),
            ctx.allocator(),
        )
    };

    let values = match bp.patches() {
        Some(patches) => {
            let mut values_mut = values.into_mut();
            apply_filtered_patches::<T>(&mut values_mut, &patches, &params, mask_values, ctx)?;
            values_mut.freeze()
        }
        None => values,
    };

    Ok(PrimitiveArray::new(values, validity).into_array())
}

fn filter_bitpacked_narrow<T, U>(
    bp: ArrayView<'_, BitPacked>,
    bit_width: usize,
    params: &Params<T>,
    mask_values: &MaskValues,
    allocator: &BufferAllocatorRef,
) -> Buffer<T>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    U: PhysicalPType<Physical = U> + BitPacking + AsPrimitive<u64>,
{
    let chunks =
        AffineNarrowBitPackedChunks::<T, U>::new(bp.packed_slice::<U>(), bit_width, params);
    filter_chunked(&chunks, params.offset, mask_values.selection(), allocator)
}

/// Apply surviving [`BitPacked`] patches to `values` after `filter_chunked`, decoding each
/// surviving patch residual with the model of its original `(chunk, j)` position.
fn apply_filtered_patches<T>(
    values: &mut [T],
    patches: &Patches,
    params: &Params<T>,
    mask_values: &MaskValues,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    let patch_indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let patch_offset = patches.offset();

    // Fast paths when patch values need no decompression: write straight into `values`.
    if let Some(constant) = patches.values().as_constant() {
        if constant.is_null() {
            return Ok(());
        }
        let patch_val = scalar_residual_as_t::<T>(constant);
        match_each_unsigned_integer_ptype!(patch_indices.ptype(), |P| {
            apply_constant_patch_values::<T, P>(
                values,
                patch_indices.as_slice::<P>(),
                patch_offset,
                patch_val,
                params,
                mask_values,
            );
        });
        return Ok(());
    }

    if let Some(prim_vals) = patches.values().as_opt::<Primitive>() {
        return match_each_unsigned_integer_ptype!(patch_indices.ptype(), |P| {
            apply_primitive_patch_values::<T, P>(
                values,
                patch_indices.as_slice::<P>(),
                patch_offset,
                prim_vals,
                params,
                mask_values,
            );
            Ok(())
        });
    }

    // Compressed patch values: find surviving slots first so sparse filters avoid decoding all
    // patches.
    let mut hits: Vec<(usize, usize, u64)> = Vec::new();
    match_each_unsigned_integer_ptype!(patch_indices.ptype(), |P| {
        collect_surviving_patch_hits::<P>(
            patch_indices.as_slice::<P>(),
            patch_offset,
            mask_values,
            &mut hits,
        );
    });
    if hits.is_empty() {
        return Ok(());
    }

    if hits.len().saturating_mul(8) < patches.num_patches() {
        let slot_indices: Buffer<u64> = hits.iter().map(|&(_, _, slot)| slot).collect();
        let taken_vals = patches
            .values()
            .take(PrimitiveArray::from_iter(slot_indices).into_array())?
            .execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(taken_vals.ptype(), |V| {
            apply_taken_patch_hits::<T, V>(values, &hits, taken_vals.as_slice::<V>(), params);
        });
    } else {
        let all_vals = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(all_vals.ptype(), |V| {
            apply_all_patch_hits::<T, V>(values, &hits, all_vals.as_slice::<V>(), params);
        });
    }
    Ok(())
}

fn apply_constant_patch_values<T, P>(
    values: &mut [T],
    patch_indices: &[P],
    patch_offset: usize,
    patch_val: T,
    params: &Params<T>,
    mask_values: &MaskValues,
) where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    P: Copy + AsPrimitive<usize>,
{
    visit_surviving_patches(
        patch_indices,
        patch_offset,
        mask_values,
        |out_idx, orig_idx, _patch_slot| {
            let pos = params.offset + orig_idx;
            values[out_idx] = params.decode(pos / FL_CHUNK_SIZE, pos % FL_CHUNK_SIZE, patch_val);
        },
    );
}

fn collect_surviving_patch_hits<P>(
    patch_indices: &[P],
    patch_offset: usize,
    mask_values: &MaskValues,
    hits: &mut Vec<(usize, usize, u64)>,
) where
    P: Copy + AsPrimitive<usize>,
{
    visit_surviving_patches(
        patch_indices,
        patch_offset,
        mask_values,
        |out_idx, orig_idx, patch_slot| {
            hits.push((out_idx, orig_idx, patch_slot as u64));
        },
    );
}

fn scalar_residual_as_t<T>(scalar: &vortex_array::scalar::Scalar) -> T
where
    T: NativePType,
    u64: AsPrimitive<T>,
{
    let prim = scalar.as_primitive();
    if prim.ptype() == T::PTYPE {
        prim.typed_value::<T>()
            .vortex_expect("patch values are non-null")
    } else {
        <u64 as AsPrimitive<T>>::as_(prim.as_::<u64>().vortex_expect("patch values are non-null"))
    }
}

fn apply_primitive_patch_values<T, P>(
    values: &mut [T],
    patch_indices: &[P],
    patch_offset: usize,
    prim_vals: ArrayView<'_, Primitive>,
    params: &Params<T>,
    mask_values: &MaskValues,
) where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    P: Copy + AsPrimitive<usize>,
{
    match_each_integer_ptype!(prim_vals.ptype(), |V| {
        apply_primitive_patch_values_typed::<T, P, V>(
            values,
            patch_indices,
            patch_offset,
            prim_vals.as_slice::<V>(),
            params,
            mask_values,
        )
    });
}

fn apply_primitive_patch_values_typed<T, P, V>(
    values: &mut [T],
    patch_indices: &[P],
    patch_offset: usize,
    patch_values: &[V],
    params: &Params<T>,
    mask_values: &MaskValues,
) where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    P: Copy + AsPrimitive<usize>,
    V: NativePType + AsPrimitive<u64>,
{
    visit_surviving_patches(
        patch_indices,
        patch_offset,
        mask_values,
        |out_idx, orig_idx, patch_slot| {
            let pos = params.offset + orig_idx;
            let residual: T = <u64 as AsPrimitive<T>>::as_(patch_values[patch_slot].as_());
            values[out_idx] = params.decode(pos / FL_CHUNK_SIZE, pos % FL_CHUNK_SIZE, residual);
        },
    );
}

fn apply_taken_patch_hits<T, V>(
    values: &mut [T],
    hits: &[(usize, usize, u64)],
    taken_values: &[V],
    params: &Params<T>,
) where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    V: NativePType + AsPrimitive<u64>,
{
    for (&(out_idx, orig_idx, _), &val) in hits.iter().zip(taken_values) {
        let pos = params.offset + orig_idx;
        let residual: T = <u64 as AsPrimitive<T>>::as_(val.as_());
        values[out_idx] = params.decode(pos / FL_CHUNK_SIZE, pos % FL_CHUNK_SIZE, residual);
    }
}

fn apply_all_patch_hits<T, V>(
    values: &mut [T],
    hits: &[(usize, usize, u64)],
    all_values: &[V],
    params: &Params<T>,
) where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    V: NativePType + AsPrimitive<u64>,
{
    for &(out_idx, orig_idx, slot) in hits {
        let pos = params.offset + orig_idx;
        let residual: T = <u64 as AsPrimitive<T>>::as_(all_values[slot as usize].as_());
        values[out_idx] = params.decode(pos / FL_CHUNK_SIZE, pos % FL_CHUNK_SIZE, residual);
    }
}

/// Calls `on_hit(out_idx, orig_idx, patch_slot)` for every patch whose array-local index is set
/// in `mask_values`.
fn visit_surviving_patches<P: Copy + AsPrimitive<usize>>(
    patch_indices: &[P],
    patch_offset: usize,
    mask_values: &MaskValues,
    mut on_hit: impl FnMut(usize, usize, usize),
) {
    let patch_count = patch_indices.len();
    let true_count = mask_values.true_count();

    if patch_count > true_count.saturating_mul(16)
        || (mask_values.cached_indices().is_some() && true_count <= patch_count)
    {
        let mut cursor = 0usize;
        for (out_idx, &orig_idx) in mask_values.indices().iter().enumerate() {
            if cursor >= patch_count {
                break;
            }
            let target = orig_idx + patch_offset;
            let mut step = 1usize;
            while cursor + step < patch_count && patch_indices[cursor + step].as_() < target {
                cursor += step;
                step <<= 1;
            }
            let hi = (cursor + step + 1).min(patch_count);
            cursor += patch_indices[cursor..hi].partition_point(|&pi| pi.as_() < target);
            if cursor < patch_count && patch_indices[cursor].as_() == target {
                on_hit(out_idx, orig_idx, cursor);
                cursor += 1;
            }
        }
    } else {
        let bits = mask_values.bit_buffer();
        let mut prev_idx = 0usize;
        let mut out_idx = 0usize;
        for (patch_slot, &pi) in patch_indices.iter().enumerate() {
            let orig_idx = pi.as_() - patch_offset;
            if bits.value(orig_idx) {
                out_idx += bits.count_range(prev_idx, orig_idx);
                prev_idx = orig_idx;
                on_hit(out_idx, orig_idx, patch_slot);
            }
        }
    }
}

/// Apply the affine model to already-filtered residuals from a non-BitPacked child.
fn filter_residuals_typed<T>(
    array: ArrayView<'_, Affine>,
    filtered_encoded: PrimitiveArray,
    mask_values: &MaskValues,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    let params = Params::<T>::execute(array, ctx)?;
    let validity = filtered_encoded.validity()?;
    let encoded_ptype = filtered_encoded.ptype();

    let values = if encoded_ptype == T::PTYPE {
        let mut values = filtered_encoded.into_buffer::<T>().into_mut();
        apply_filtered_in_place(&mut values, &params, mask_values);
        values.freeze()
    } else {
        match_each_unsigned_integer_ptype!(encoded_ptype, |U| {
            apply_filtered_narrow::<T, U>(
                filtered_encoded.as_slice::<U>(),
                &params,
                mask_values,
                ctx.allocator(),
            )
        })
    };

    Ok(PrimitiveArray::new(values, validity).into_array())
}

fn apply_filtered_in_place<T>(values: &mut [T], params: &Params<T>, mask_values: &MaskValues)
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
{
    if params.all_constant_flat() {
        params.apply_chunk(0, 0, values);
        return;
    }

    if params.slopes_are_zero() {
        apply_filtered_flat_chunks(mask_values, params.offset, |chunk_idx, out_start, count| {
            params.apply_chunk(chunk_idx, 0, &mut values[out_start..out_start + count]);
        });
        return;
    }

    let offset = params.offset;
    let shift = params.shift;
    visit_mask_chunks(
        mask_values,
        offset,
        |chunk_idx, out_start, chunk_positions| {
            let reference = params.references.get(chunk_idx);
            let scale = params.scales.get(chunk_idx);
            let slope = params.slopes.get(chunk_idx);
            if slope == 0 {
                let count = chunk_positions.len();
                params.apply_chunk(chunk_idx, 0, &mut values[out_start..out_start + count]);
            } else {
                let (whole, frac) = split_slope(slope, shift);
                for (val, &j) in values[out_start..].iter_mut().zip(chunk_positions) {
                    let term: T = slope_term_split(whole, frac, j, shift).as_();
                    *val = val
                        .wrapping_mul(&scale)
                        .wrapping_add(&reference)
                        .wrapping_add(&term);
                }
            }
        },
    );
}

fn apply_filtered_narrow<T, U>(
    src: &[U],
    params: &Params<T>,
    mask_values: &MaskValues,
    allocator: &BufferAllocatorRef,
) -> Buffer<T>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    U: Copy + AsPrimitive<u64>,
{
    let len = src.len();
    let mut out = BufferMut::<T>::with_capacity_in(len, allocator.clone());
    let dst = &mut out.spare_capacity_mut()[..len];

    if params.all_constant_flat() {
        params.apply_narrow(0, 0, src, dst);
    } else if params.slopes_are_zero() {
        apply_filtered_flat_chunks(mask_values, params.offset, |chunk_idx, out_start, count| {
            params.apply_narrow(
                chunk_idx,
                0,
                &src[out_start..out_start + count],
                &mut dst[out_start..out_start + count],
            );
        });
    } else {
        visit_mask_chunks(
            mask_values,
            params.offset,
            |chunk_idx, out_start, positions| {
                let count = positions.len();
                params.apply_selected_narrow(
                    chunk_idx,
                    positions,
                    &src[out_start..out_start + count],
                    &mut dst[out_start..out_start + count],
                );
            },
        );
    }

    // SAFETY: every element in `0..len` was initialized above.
    unsafe { out.set_len(len) };
    out.freeze()
}

/// Calls `on_chunk(chunk_idx, out_start, count)` for each chunk containing at least one selected
/// row, using fast word-level popcount when the mask is a bitmap.
fn apply_filtered_flat_chunks(
    mask_values: &MaskValues,
    offset: usize,
    mut on_chunk: impl FnMut(usize, usize, usize),
) {
    if let Some(indices) = mask_values.cached_indices() {
        let mut out_start = 0usize;
        let mut rest = indices;
        while let Some(&first) = rest.first() {
            let chunk_idx = (first + offset) / FL_CHUNK_SIZE;
            let chunk_end = (chunk_idx + 1) * FL_CHUNK_SIZE - offset;
            let count = rest.partition_point(|&i| i < chunk_end);
            on_chunk(chunk_idx, out_start, count);
            out_start += count;
            rest = &rest[count..];
        }
        return;
    }

    let bits = mask_values.bit_buffer();
    let len = bits.len();
    let num_chunks = (offset + len).div_ceil(FL_CHUNK_SIZE);
    let mut out_start = 0usize;
    for chunk_idx in 0..num_chunks {
        let chunk_start = chunk_idx * FL_CHUNK_SIZE;
        let start = chunk_start.saturating_sub(offset);
        let end = (chunk_start + FL_CHUNK_SIZE - offset).min(len);
        let count = bits.count_range(start, end);
        if count > 0 {
            on_chunk(chunk_idx, out_start, count);
            out_start += count;
        }
    }
}

/// Visits selected in-chunk positions grouped by chunk, passing `(chunk_idx, out_start, positions)`
/// to `on_chunk`.
fn visit_mask_chunks(
    mask_values: &MaskValues,
    offset: usize,
    mut on_chunk: impl FnMut(usize, usize, &[usize]),
) {
    let mut positions = [0usize; FL_CHUNK_SIZE];
    if let Some(indices) = mask_values.cached_indices() {
        let mut out_start = 0usize;
        let mut rest = indices;
        while let Some(&first) = rest.first() {
            let chunk_idx = (first + offset) / FL_CHUNK_SIZE;
            let chunk_start = chunk_idx * FL_CHUNK_SIZE;
            let chunk_end = chunk_start + FL_CHUNK_SIZE - offset;
            let count = rest.partition_point(|&i| i < chunk_end);
            for (dst, &orig_idx) in positions[..count].iter_mut().zip(&rest[..count]) {
                *dst = orig_idx + offset - chunk_start;
            }
            on_chunk(chunk_idx, out_start, &positions[..count]);
            out_start += count;
            rest = &rest[count..];
        }
        return;
    }

    let bits = mask_values.bit_buffer();
    let len = bits.len();
    let num_chunks = (offset + len).div_ceil(FL_CHUNK_SIZE);
    let mut out_start = 0usize;
    for chunk_idx in 0..num_chunks {
        let chunk_start = chunk_idx * FL_CHUNK_SIZE;
        let start = chunk_start.saturating_sub(offset);
        let end = (chunk_start + FL_CHUNK_SIZE - offset).min(len);
        let chunk_offset = start + offset - chunk_start;
        let mut count = 0usize;
        bits.slice(start..end).for_each_set_index(|rel| {
            positions[count] = chunk_offset + rel;
            count += 1;
        });
        if count > 0 {
            on_chunk(chunk_idx, out_start, &positions[..count]);
            out_start += count;
        }
    }
}

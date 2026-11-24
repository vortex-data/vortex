// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chunk-pruned `take` kernel for [`Affine`].

use std::mem::MaybeUninit;

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
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::AffineNarrowBitPackedChunks;
use super::AffineSameWidthBitPackedChunks;
use super::resolve_aligned_bitpacked;
use super::unpack_chunk_threshold;
use crate::Affine;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::affine::array::AffineArrayExt;
use crate::affine::array::AffineArraySlotsExt;
use crate::affine::array::affine_decompress::Params;
use crate::take_chunks::check_indices;
use crate::take_chunks::nullable_positions;
use crate::take_chunks::take_in_order;
use crate::take_chunks::valid_positions;

/// Above this fraction of the array, fused full-array decode plus primitive gather beats sparse
/// chunked take.
const FULL_ARRAY_DECODE_RATIO: usize = 8;

/// With dense patches (> 1/256 rows patched), full-array decode + sequential patch application
/// beats per-index patch search once take selectivity exceeds 1/16.
const PATCHED_FULL_DECODE_RATIO: usize = 16;
const DENSE_PATCH_RATIO: usize = 256;

impl TakeExecute for Affine {
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

        if let Some(bp_arr) = resolve_aligned_bitpacked(array, ctx)? {
            let bp = bp_arr.as_::<BitPacked>();
            let bit_width = bp.bit_width();
            if let Some(patches) = bp.patches()
                && indices.len().saturating_mul(PATCHED_FULL_DECODE_RATIO) > array.len()
                && patches.num_patches().saturating_mul(DENSE_PATCH_RATIO) > array.len()
            {
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
            let taken_validity = array.validity()?.take(&indices_prim.clone().into_array())?;
            return match_each_integer_ptype!(array.ptype(), |T| {
                take_bitpacked_typed::<T>(
                    array,
                    bp,
                    bit_width,
                    &indices_prim,
                    &index_validity,
                    taken_validity,
                    ctx,
                )
                .map(Some)
            });
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
        let taken_encoded = array
            .encoded()
            .take(indices_prim.clone().into_array())?
            .execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(array.ptype(), |T| {
            take_residuals_typed::<T>(array, taken_encoded, &indices_prim, &index_validity, ctx)
                .map(Some)
        })
    }
}

fn take_bitpacked_typed<T>(
    array: ArrayView<'_, Affine>,
    bp: ArrayView<'_, BitPacked>,
    bit_width: u8,
    indices: &PrimitiveArray,
    index_validity: &Mask,
    taken_validity: Validity,
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
    match_each_integer_ptype!(indices.ptype(), |I| {
        take_bitpacked_with_index::<T, I>(
            array,
            bp,
            bit_width,
            indices,
            index_validity,
            taken_validity,
            ctx,
        )
    })
}

fn take_bitpacked_with_index<T, I>(
    array: ArrayView<'_, Affine>,
    bp: ArrayView<'_, BitPacked>,
    bit_width: u8,
    indices: &PrimitiveArray,
    index_validity: &Mask,
    taken_validity: Validity,
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
    I: IntegerPType,
{
    let indices_slice = indices.as_slice::<I>();
    check_indices(indices_slice, index_validity, array.len())?;

    let params = Params::<T>::execute(array, ctx)?;
    let encoded_ptype = bp.dtype().as_ptype();
    let bit_width = usize::from(bit_width);
    let sparse_limit = unpack_chunk_threshold::<T>();
    let mut output = BufferMut::<T>::with_capacity_in(indices.len(), ctx.allocator().clone());
    let mut scratch = [const { MaybeUninit::<T>::uninit() }; FL_CHUNK_SIZE];

    if encoded_ptype.byte_width() < T::PTYPE.byte_width() {
        match_each_unsigned_integer_ptype!(encoded_ptype, |U| {
            take_bitpacked_narrow::<T, U, I>(
                bp,
                bit_width,
                &params,
                indices_slice,
                index_validity,
                sparse_limit,
                &mut output,
                &mut scratch,
            )
        });
    } else {
        let chunks = AffineSameWidthBitPackedChunks::<T>::new(
            bp.packed_slice::<T::Physical>(),
            bit_width,
            &params,
        );
        match index_validity {
            Mask::AllTrue(_) => take_in_order(
                &chunks,
                &mut output,
                valid_positions(indices_slice, params.offset),
                sparse_limit,
                &mut scratch,
            ),
            _ => take_in_order(
                &chunks,
                &mut output,
                nullable_positions(indices_slice, index_validity, params.offset),
                sparse_limit,
                &mut scratch,
            ),
        }
    }

    if let Some(patches) = bp.patches() {
        apply_taken_patches::<T, I>(&mut output, &patches, indices, indices_slice, &params, ctx)?;
    }

    Ok(PrimitiveArray::new(output.freeze(), taken_validity).into_array())
}

#[expect(
    clippy::too_many_arguments,
    reason = "typed bitpacked take dispatch parameters"
)]
fn take_bitpacked_narrow<T, U, I>(
    bp: ArrayView<'_, BitPacked>,
    bit_width: usize,
    params: &Params<T>,
    indices_slice: &[I],
    index_validity: &Mask,
    sparse_limit: usize,
    output: &mut BufferMut<T>,
    scratch: &mut [MaybeUninit<T>; FL_CHUNK_SIZE],
) where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    U: PhysicalPType<Physical = U> + BitPacking + AsPrimitive<u64>,
    I: IntegerPType,
{
    let chunks =
        AffineNarrowBitPackedChunks::<T, U>::new(bp.packed_slice::<U>(), bit_width, params);
    match index_validity {
        Mask::AllTrue(_) => take_in_order(
            &chunks,
            output,
            valid_positions(indices_slice, params.offset),
            sparse_limit,
            scratch,
        ),
        _ => take_in_order(
            &chunks,
            output,
            nullable_positions(indices_slice, index_validity, params.offset),
            sparse_limit,
            scratch,
        ),
    }
}

/// Apply [`BitPacked`] patches to `output` for the taken `indices`, decoding each matched patch
/// residual with the model of its original `(chunk, j)` position.
fn apply_taken_patches<T, I>(
    output: &mut [T],
    patches: &Patches,
    indices: &PrimitiveArray,
    indices_slice: &[I],
    params: &Params<T>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    I: IntegerPType,
{
    let taken_patches = if patches.chunk_offsets().is_some() || patches.num_patches() > 4096 {
        patches.take_search(indices.clone(), false, ctx)?
    } else {
        patches.take(&indices.clone().into_array(), ctx)?
    };
    let Some(taken_patches) = taken_patches else {
        return Ok(());
    };

    let out_indices = taken_patches
        .indices()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let patch_values = taken_patches
        .values()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let patch_offset = taken_patches.offset();

    match_each_unsigned_integer_ptype!(out_indices.ptype(), |P| {
        apply_taken_patches_typed::<T, I, P>(
            output,
            out_indices.as_slice::<P>(),
            patch_offset,
            &patch_values,
            indices_slice,
            params,
        )
    });
    Ok(())
}

fn apply_taken_patches_typed<T, I, P>(
    output: &mut [T],
    out_indices: &[P],
    patch_offset: usize,
    patch_values: &PrimitiveArray,
    indices_slice: &[I],
    params: &Params<T>,
) where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    I: IntegerPType,
    P: Copy + AsPrimitive<usize>,
{
    match_each_integer_ptype!(patch_values.ptype(), |V| {
        apply_taken_patches_values::<T, I, P, V>(
            output,
            out_indices,
            patch_offset,
            patch_values.as_slice::<V>(),
            indices_slice,
            params,
        )
    });
}

fn apply_taken_patches_values<T, I, P, V>(
    output: &mut [T],
    out_indices: &[P],
    patch_offset: usize,
    patch_values: &[V],
    indices_slice: &[I],
    params: &Params<T>,
) where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    I: IntegerPType,
    P: Copy + AsPrimitive<usize>,
    V: NativePType + AsPrimitive<u64>,
{
    for (&out_idx, &val) in out_indices.iter().zip(patch_values) {
        let out_idx = out_idx.as_() - patch_offset;
        let orig_idx = AsPrimitive::<usize>::as_(indices_slice[out_idx]);
        let pos = params.offset + orig_idx;
        let residual: T = <u64 as AsPrimitive<T>>::as_(val.as_());
        output[out_idx] = params.decode(pos / FL_CHUNK_SIZE, pos % FL_CHUNK_SIZE, residual);
    }
}

/// Apply the affine model to already-taken residuals from a non-BitPacked child.
fn take_residuals_typed<T>(
    array: ArrayView<'_, Affine>,
    taken_encoded: PrimitiveArray,
    indices: &PrimitiveArray,
    index_validity: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    let params = Params::<T>::execute(array, ctx)?;
    let validity = taken_encoded.validity()?;
    let encoded_ptype = taken_encoded.ptype();

    let values = if encoded_ptype == T::PTYPE {
        let mut values = taken_encoded.into_buffer::<T>().into_mut();
        match_each_integer_ptype!(indices.ptype(), |I| {
            apply_taken_in_place::<T, I>(
                &mut values,
                indices.as_slice::<I>(),
                index_validity,
                &params,
            )
        });
        values.freeze()
    } else {
        match_each_unsigned_integer_ptype!(encoded_ptype, |U| {
            take_residuals_narrow_index::<T, U>(
                taken_encoded.as_slice::<U>(),
                indices,
                index_validity,
                &params,
                ctx.allocator(),
            )
        })
    };

    Ok(PrimitiveArray::new(values, validity).into_array())
}

fn take_residuals_narrow_index<T, U>(
    src: &[U],
    indices: &PrimitiveArray,
    index_validity: &Mask,
    params: &Params<T>,
    allocator: &BufferAllocatorRef,
) -> Buffer<T>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    U: Copy + AsPrimitive<u64>,
{
    match_each_integer_ptype!(indices.ptype(), |I| {
        apply_taken_narrow::<T, U, I>(
            src,
            indices.as_slice::<I>(),
            index_validity,
            params,
            allocator,
        )
    })
}

fn apply_taken_in_place<T, I>(
    values: &mut [T],
    indices: &[I],
    index_validity: &Mask,
    params: &Params<T>,
) where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    I: IntegerPType,
{
    if params.all_constant_flat() {
        params.apply_chunk(0, 0, values);
        return;
    }

    let offset = params.offset;
    match index_validity {
        Mask::AllTrue(_) => {
            for (val, &idx) in values.iter_mut().zip(indices) {
                let pos = offset + AsPrimitive::<usize>::as_(idx);
                *val = params.decode(pos / FL_CHUNK_SIZE, pos % FL_CHUNK_SIZE, *val);
            }
        }
        Mask::AllFalse(_) => values.fill(T::zero()),
        Mask::Values(mask_vals) => {
            let bits = mask_vals.bit_buffer();
            for (i, (val, &idx)) in values.iter_mut().zip(indices).enumerate() {
                if bits.value(i) {
                    let pos = offset + AsPrimitive::<usize>::as_(idx);
                    *val = params.decode(pos / FL_CHUNK_SIZE, pos % FL_CHUNK_SIZE, *val);
                } else {
                    *val = T::zero();
                }
            }
        }
    }
}

fn apply_taken_narrow<T, U, I>(
    src: &[U],
    indices: &[I],
    index_validity: &Mask,
    params: &Params<T>,
    allocator: &BufferAllocatorRef,
) -> Buffer<T>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    U: Copy + AsPrimitive<u64>,
    I: IntegerPType,
{
    let len = src.len();
    let mut out = BufferMut::<T>::with_capacity_in(len, allocator.clone());
    let dst = &mut out.spare_capacity_mut()[..len];

    if params.all_constant_flat() {
        params.apply_narrow(0, 0, src, dst);
    } else {
        let offset = params.offset;
        match index_validity {
            Mask::AllTrue(_) => {
                for ((slot, &e), &idx) in dst.iter_mut().zip(src).zip(indices) {
                    let pos = offset + AsPrimitive::<usize>::as_(idx);
                    let residual: T = <u64 as AsPrimitive<T>>::as_(e.as_());
                    slot.write(params.decode(pos / FL_CHUNK_SIZE, pos % FL_CHUNK_SIZE, residual));
                }
            }
            Mask::AllFalse(_) => {
                for slot in dst.iter_mut() {
                    slot.write(T::zero());
                }
            }
            Mask::Values(mask_vals) => {
                let bits = mask_vals.bit_buffer();
                for (i, ((slot, &e), &idx)) in dst.iter_mut().zip(src).zip(indices).enumerate() {
                    if bits.value(i) {
                        let pos = offset + AsPrimitive::<usize>::as_(idx);
                        let residual: T = <u64 as AsPrimitive<T>>::as_(e.as_());
                        slot.write(params.decode(
                            pos / FL_CHUNK_SIZE,
                            pos % FL_CHUNK_SIZE,
                            residual,
                        ));
                    } else {
                        slot.write(T::zero());
                    }
                }
            }
        }
    }

    // SAFETY: every slot in `0..len` was initialized above.
    unsafe { out.set_len(len) };
    out.freeze()
}

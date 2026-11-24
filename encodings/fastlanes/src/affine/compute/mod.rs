// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chunk-level compute kernels (`filter`, `take`, `compare`) for [`Affine`].

mod compare;
mod filter;
mod take;

use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use fastlanes::FoR;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingAdd;
use num_traits::WrappingMul;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Slice;
use vortex_array::arrays::slice::SliceArraySlotsExt;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_buffer::filter::ChunkDecoder;
use vortex_buffer::filter::FILTER_CHUNK_LEN;
use vortex_error::VortexResult;

use crate::Affine;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::affine::array::AffineArrayExt;
use crate::affine::array::AffineArraySlotsExt;
use crate::affine::array::affine_decompress::Params;
use crate::affine::array::affine_decompress::physical_as_logical;

const _: () = assert!(FL_CHUNK_SIZE == FILTER_CHUNK_LEN);

/// Maximum number of indices for which [`AffineNarrowBitPackedChunks::decode_indices`] extracts
/// individual values instead of unpacking the whole narrow chunk first.
pub(super) const fn unpack_chunk_threshold<U>() -> usize {
    match size_of::<U>() {
        1 => 16,
        2 => 32,
        4 => 64,
        8 => 160,
        _ => unreachable!(),
    }
}

/// Maximum sparse index count supported by stack-allocated scratch in `decode_indices`.
const MAX_SPARSE_STACK: usize = unpack_chunk_threshold::<u64>();

/// Resolves `array.encoded()` to an aligned [`BitPacked`] array when possible.
///
/// Slicing a patched `BitPacked` child leaves a lazy `Slice` wrapper; resolving it via
/// [`SliceKernel`] keeps the fused chunk-pruned kernels active on sliced arrays.
pub(super) fn resolve_aligned_bitpacked(
    array: ArrayView<'_, Affine>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let encoded = array.encoded();
    let candidate = if let Some(bp) = encoded.as_opt::<BitPacked>() {
        Some(bp.array().clone())
    } else if let Some(slice) = encoded.as_opt::<Slice>()
        && let Some(bp) = slice.child().as_opt::<BitPacked>()
    {
        <BitPacked as SliceKernel>::slice(bp, slice.slice_range().clone(), ctx)?
    } else {
        None
    };
    Ok(candidate.filter(|arr| {
        arr.as_opt::<BitPacked>()
            .is_some_and(|bp| bp.offset() == array.offset())
    }))
}

/// Chunk decoder for an [`Affine`] array backed by an aligned [`BitPacked`] child whose physical
/// residual type `U` is narrower than `T`.
pub(super) struct AffineNarrowBitPackedChunks<'a, T, U> {
    packed: &'a [U],
    bit_width: usize,
    packed_chunk_len: usize,
    params: &'a Params<T>,
}

impl<'a, T, U> AffineNarrowBitPackedChunks<'a, T, U> {
    pub(super) fn new(packed: &'a [U], bit_width: usize, params: &'a Params<T>) -> Self {
        Self {
            packed,
            bit_width,
            packed_chunk_len: FL_CHUNK_SIZE * bit_width / (8 * size_of::<U>()),
            params,
        }
    }

    #[inline]
    fn chunk(&self, chunk_idx: usize) -> &[U] {
        &self.packed[chunk_idx * self.packed_chunk_len..][..self.packed_chunk_len]
    }
}

// SAFETY: Both `decode_chunk` and `decode_indices` initialize every element of `dst`.
unsafe impl<T, U> ChunkDecoder<T> for AffineNarrowBitPackedChunks<'_, T, U>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    U: PhysicalPType<Physical = U> + BitPacking + AsPrimitive<u64>,
{
    #[inline]
    fn decode_chunk(&self, chunk_idx: usize, dst: &mut [MaybeUninit<T>; FILTER_CHUNK_LEN]) {
        let mut scratch = [U::default(); FL_CHUNK_SIZE];
        // SAFETY: `self.chunk(chunk_idx)` holds one chunk packed at `self.bit_width` bits.
        unsafe {
            BitPacking::unchecked_unpack(self.bit_width, self.chunk(chunk_idx), &mut scratch);
        }
        self.params.apply_narrow(chunk_idx, 0, &scratch, dst);
    }

    #[inline]
    fn decode_indices(&self, chunk_idx: usize, indices: &[usize], dst: &mut [MaybeUninit<T>]) {
        debug_assert_eq!(indices.len(), dst.len());
        debug_assert!(indices.iter().all(|&index| index < FL_CHUNK_SIZE));
        let count = indices.len();
        if count == 0 {
            return;
        }
        let packed_chunk = self.chunk(chunk_idx);
        if count <= unpack_chunk_threshold::<U>() {
            let mut unpacked = [const { MaybeUninit::<U>::uninit() }; MAX_SPARSE_STACK];
            // SAFETY: every index is < FL_CHUNK_SIZE and `unpacked[..count]` holds `count` slots.
            unsafe {
                BitPacking::unchecked_unpack_indices(
                    self.bit_width,
                    packed_chunk,
                    indices,
                    &mut unpacked[..count],
                );
            }
            // SAFETY: `unchecked_unpack_indices` initialized the first `count` slots.
            let unpacked = unsafe { unpacked[..count].assume_init_ref() };
            self.params
                .apply_selected_narrow(chunk_idx, indices, unpacked, dst);
        } else {
            let mut scratch = [U::default(); FL_CHUNK_SIZE];
            // SAFETY: `packed_chunk` holds one chunk packed at `self.bit_width` bits.
            unsafe {
                BitPacking::unchecked_unpack(self.bit_width, packed_chunk, &mut scratch);
            }
            self.params
                .apply_gather_narrow(chunk_idx, indices, &scratch, dst);
        }
    }
}

/// Chunk decoder for an [`Affine`] array backed by an aligned [`BitPacked`] child of the same
/// width as `T`.
///
/// Flat chunks (`scale == 1 && slope == 0`) unpack straight into `dst` with FastLanes' fused
/// `FoR::unchecked_unfor_pack`.
pub(super) struct AffineSameWidthBitPackedChunks<'a, T: PhysicalPType> {
    packed: &'a [T::Physical],
    bit_width: usize,
    packed_chunk_len: usize,
    params: &'a Params<T>,
}

impl<'a, T: PhysicalPType> AffineSameWidthBitPackedChunks<'a, T> {
    pub(super) fn new(packed: &'a [T::Physical], bit_width: usize, params: &'a Params<T>) -> Self {
        Self {
            packed,
            bit_width,
            packed_chunk_len: FL_CHUNK_SIZE * bit_width / (8 * size_of::<T::Physical>()),
            params,
        }
    }

    #[inline]
    fn chunk(&self, chunk_idx: usize) -> &[T::Physical] {
        &self.packed[chunk_idx * self.packed_chunk_len..][..self.packed_chunk_len]
    }
}

// SAFETY: Both `decode_chunk` and `decode_indices` initialize every element of `dst`.
unsafe impl<T> ChunkDecoder<T> for AffineSameWidthBitPackedChunks<'_, T>
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
    #[inline]
    fn decode_chunk(&self, chunk_idx: usize, dst: &mut [MaybeUninit<T>; FILTER_CHUNK_LEN]) {
        let packed_chunk = self.chunk(chunk_idx);
        let mut scratch = [T::Physical::default(); FL_CHUNK_SIZE];
        if self.params.is_flat(chunk_idx) {
            let reference = self.params.references.get(chunk_idx).as_();
            // SAFETY: `packed_chunk` holds one chunk packed at `self.bit_width` bits.
            unsafe {
                FoR::unchecked_unfor_pack(self.bit_width, packed_chunk, reference, &mut scratch)
            };
        } else {
            // SAFETY: `packed_chunk` holds one chunk packed at `self.bit_width` bits.
            unsafe { BitPacking::unchecked_unpack(self.bit_width, packed_chunk, &mut scratch) };
            self.params
                .apply_chunk(chunk_idx, 0, physical_as_logical::<T>(&mut scratch));
        }
        let logical = physical_as_logical::<T>(&mut scratch);
        for (out, &val) in dst.iter_mut().zip(logical.iter()) {
            out.write(val);
        }
    }

    #[inline]
    fn decode_indices(&self, chunk_idx: usize, indices: &[usize], dst: &mut [MaybeUninit<T>]) {
        debug_assert_eq!(indices.len(), dst.len());
        debug_assert!(indices.iter().all(|&index| index < FL_CHUNK_SIZE));
        let count = indices.len();
        if count == 0 {
            return;
        }
        let packed_chunk = self.chunk(chunk_idx);
        if count <= unpack_chunk_threshold::<T::Physical>() {
            let mut unpacked = [const { MaybeUninit::<T::Physical>::uninit() }; MAX_SPARSE_STACK];
            // SAFETY: every index is < FL_CHUNK_SIZE and `unpacked[..count]` holds `count` slots.
            unsafe {
                BitPacking::unchecked_unpack_indices(
                    self.bit_width,
                    packed_chunk,
                    indices,
                    &mut unpacked[..count],
                );
            }
            // SAFETY: `unchecked_unpack_indices` initialized the first `count` slots.
            let unpacked = unsafe { unpacked[..count].assume_init_ref() };
            self.params
                .apply_selected_narrow(chunk_idx, indices, unpacked, dst);
        } else {
            let mut scratch = [T::Physical::default(); FL_CHUNK_SIZE];
            // SAFETY: `packed_chunk` holds one chunk packed at `self.bit_width` bits.
            unsafe {
                BitPacking::unchecked_unpack(self.bit_width, packed_chunk, &mut scratch);
            }
            self.params
                .apply_gather_narrow(chunk_idx, indices, &scratch, dst);
        }
    }
}

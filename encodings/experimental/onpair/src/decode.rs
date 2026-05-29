// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Bridge between [`OnPair`] slot children and the upstream `onpair` crate's
//! decompression API.

use std::mem::MaybeUninit;

use num_traits::AsPrimitive;
use onpair::Parts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArraySlotsExt;

/// Materialised, host-resident copies of every read path's input.
pub struct OwnedDecodeInputs {
    pub dict_bytes: ByteBuffer,
    pub dict_offsets: Buffer<u32>,
    pub codes: Buffer<u16>,
    pub code_boundaries: Buffer<u32>,
    pub bits: u32,
}

/// Canonicalise a slot child to a `PrimitiveArray` (decoding any cascading
/// encoding the compressor chose — Delta, FastLanes bit-pack, narrowing — to
/// absolute primitive values), then widen element-wise to the decoder's
/// native width `T`.
///
/// Going through `cast(dtype).execute()` is unsafe here: the `Delta` cast
/// kernel preserves the Delta wrapping and only widens the inner bases/deltas,
/// but the fastlanes bases-per-chunk layout is keyed on LANES (e.g. u8 → 64,
/// u32 → 16), so the widened Delta decodes against misaligned bases and
/// produces non-monotonic offsets.
fn collect_widened<T: NativePType>(
    arr: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>>
where
    u8: AsPrimitive<T>,
    i8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    i16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    i32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    i64: AsPrimitive<T>,
{
    let prim = arr.clone().execute::<PrimitiveArray>(ctx)?;
    if prim.ptype() == T::PTYPE {
        return Ok(prim.into_buffer::<T>());
    }
    Ok(match_each_integer_ptype!(prim.ptype(), |P| {
        let slice = prim.as_slice::<P>();
        let mut out = BufferMut::<T>::with_capacity(slice.len());
        for &v in slice {
            // SAFETY: capacity reserved above.
            unsafe { out.push_unchecked(v.as_()) };
        }
        out.freeze()
    }))
}

impl OwnedDecodeInputs {
    pub fn collect(array: ArrayView<'_, OnPair>, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Ok(Self {
            dict_bytes: array.dict_bytes().clone(),
            dict_offsets: collect_widened::<u32>(array.dict_offsets(), ctx)?,
            codes: collect_widened::<u16>(array.codes(), ctx)?,
            code_boundaries: collect_widened::<u32>(array.codes_offsets(), ctx)?,
            bits: array.bits(),
        })
    }

    /// Total decoded byte length across all rows.
    #[inline]
    pub fn decompressed_len(&self) -> usize {
        onpair::decompressed_len(self.as_parts())
    }

    /// Decoded byte length of a single row.
    #[inline]
    pub fn decompressed_row_len(&self, row: usize) -> usize {
        onpair::decompressed_row_len(self.as_parts(), row)
    }

    /// Decode every row contiguously into `out`. Returns the number of
    /// initialised bytes.
    #[inline]
    pub fn decompress_into(&self, out: &mut [MaybeUninit<u8>]) -> usize {
        onpair::decompress_into(self.as_parts(), out)
    }

    /// Decode a single row into `out`. Returns the number of initialised
    /// bytes.
    #[inline]
    pub fn decompress_row_into(&self, row: usize, out: &mut [MaybeUninit<u8>]) -> usize {
        onpair::decompress_row_into(self.as_parts(), row, out)
    }

    fn as_parts(&self) -> Parts<'_, u32> {
        Parts {
            dict_bytes: self.dict_bytes.as_slice(),
            dict_offsets: self.dict_offsets.as_slice(),
            bits: self.bits,
            codes: self.codes.as_slice(),
            code_boundaries: self.code_boundaries.as_slice(),
        }
    }
}

/// Inputs for whole-column decompression.
///
/// Unlike [`OwnedDecodeInputs`], this deliberately omits the per-row
/// `code_boundaries` (`codes_offsets`) child: the contiguous
/// [`onpair::decompress_into`] decoder walks the flat `codes` stream directly
/// and never consults the per-row boundaries. Materialising that child for a
/// full canonicalisation is pure overhead — for a narrowed/bit-packed
/// `codes_offsets` it also forces an extra child `execute`.
pub struct FullDecodeInputs {
    dict_bytes: ByteBuffer,
    dict_offsets: Buffer<u32>,
    codes: Buffer<u16>,
    bits: u32,
}

impl FullDecodeInputs {
    pub fn collect(array: ArrayView<'_, OnPair>, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Ok(Self {
            dict_bytes: array.dict_bytes().clone(),
            dict_offsets: collect_widened::<u32>(array.dict_offsets(), ctx)?,
            codes: collect_widened::<u16>(array.codes(), ctx)?,
            bits: array.bits(),
        })
    }

    /// Decode every row contiguously into `out`. Returns the number of
    /// initialised bytes.
    #[inline]
    pub fn decompress_into(&self, out: &mut [MaybeUninit<u8>]) -> usize {
        onpair::decompress_into(self.as_parts(), out)
    }

    fn as_parts(&self) -> Parts<'_, u32> {
        Parts {
            dict_bytes: self.dict_bytes.as_slice(),
            dict_offsets: self.dict_offsets.as_slice(),
            bits: self.bits,
            codes: self.codes.as_slice(),
            // `decompress_into` never reads the per-row boundaries; an empty
            // slice keeps the `Parts` well-typed without materialising them.
            code_boundaries: &[],
        }
    }
}

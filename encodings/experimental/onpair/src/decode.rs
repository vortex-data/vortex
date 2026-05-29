// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Bridge between [`OnPair`] slot children and the upstream `onpair` crate's
//! decompression API.

use std::mem::MaybeUninit;
use std::ops::Range;

use onpair::Parts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_buffer::Buffer;
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

/// Canonicalise a slot child to the decoder's native primitive width.
fn collect_widened<T: NativePType>(
    arr: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>> {
    let dtype = DType::Primitive(T::PTYPE, arr.dtype().nullability());
    Ok(arr
        .cast(dtype)?
        .execute::<PrimitiveArray>(ctx)?
        .into_buffer::<T>())
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

    /// Decode a contiguous code window into `out`. Returns the number of
    /// initialised bytes.
    #[inline]
    pub fn decompress_code_range_into(
        &self,
        range: Range<usize>,
        out: &mut [MaybeUninit<u8>],
    ) -> usize {
        onpair::decompress_into(
            Parts::<u32> {
                dict_bytes: self.dict_bytes.as_slice(),
                dict_offsets: self.dict_offsets.as_slice(),
                bits: self.bits,
                codes: &self.codes.as_slice()[range],
                code_boundaries: &[],
            },
            out,
        )
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

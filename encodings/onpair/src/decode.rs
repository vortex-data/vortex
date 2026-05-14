// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Pure-Rust decoder for an [`OnPair`][crate::OnPair] array.
//!
//! Given the materialised slot children (dictionary blob + offsets +
//! per-token `codes` + per-row `codes_offsets`), every read path here is a
//! straight Rust loop — no C++, no FFI, no bit-unpacking (the codes were
//! unpacked at compress time and stored as u16).

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::OnPair;
use crate::OnPairArrayExt;

/// Materialised, host-resident copy of every read path's input.
///
/// The cascading compressor may narrow our `u16` `codes` and `u32` offset
/// children down to a tighter integer type (e.g. `u8` codes for dict-8
/// data). We widen each back to its canonical width at materialisation time
/// so the decode loop can index without per-token branching.
pub(crate) struct OwnedDecodeInputs {
    pub dict_bytes: ByteBuffer,
    pub dict_offsets: Buffer<u32>,
    pub codes: Buffer<u16>,
    pub codes_offsets: Buffer<u32>,
}

impl OwnedDecodeInputs {
    pub fn collect(array: ArrayView<'_, OnPair>, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Ok(Self {
            dict_bytes: array.dict_bytes().clone(),
            dict_offsets: widen_to_u32(array.dict_offsets(), ctx)?,
            codes: widen_to_u16(array.codes(), ctx)?,
            codes_offsets: widen_to_u32(array.codes_offsets(), ctx)?,
        })
    }

    pub fn view(&self) -> DecodeView<'_> {
        DecodeView {
            dict_bytes: self.dict_bytes.as_slice(),
            dict_offsets: self.dict_offsets.as_slice(),
            codes: self.codes.as_slice(),
            codes_offsets: self.codes_offsets.as_slice(),
        }
    }
}

fn widen_to_u16(arr: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Buffer<u16>> {
    let primitive = arr.clone().execute::<PrimitiveArray>(ctx)?;
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let widened: Buffer<u16> = match_each_integer_ptype!(primitive.ptype(), |P| {
        primitive.as_slice::<P>().iter().map(|x| *x as u16).collect()
    });
    Ok(widened)
}

fn widen_to_u32(arr: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Buffer<u32>> {
    let primitive = arr.clone().execute::<PrimitiveArray>(ctx)?;
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let widened: Buffer<u32> = match_each_integer_ptype!(primitive.ptype(), |P| {
        primitive.as_slice::<P>().iter().map(|x| *x as u32).collect()
    });
    if widened.is_empty() {
        return Err(vortex_err!("OnPair: empty offsets after widening"));
    }
    Ok(widened)
}

/// Borrowed slices for the decode loop.
#[derive(Copy, Clone)]
pub(crate) struct DecodeView<'a> {
    pub dict_bytes: &'a [u8],
    pub dict_offsets: &'a [u32],
    pub codes: &'a [u16],
    pub codes_offsets: &'a [u32],
}

impl<'a> DecodeView<'a> {
    /// Decode row `row` into `out` (appended).
    #[inline]
    pub fn decode_row_into(&self, row: usize, out: &mut Vec<u8>) {
        let lo = self.codes_offsets[row] as usize;
        let hi = self.codes_offsets[row + 1] as usize;
        for &c in &self.codes[lo..hi] {
            let dlo = self.dict_offsets[c as usize] as usize;
            let dhi = self.dict_offsets[c as usize + 1] as usize;
            out.extend_from_slice(&self.dict_bytes[dlo..dhi]);
        }
    }

    /// Decoded byte length of row `row` without actually copying bytes.
    #[inline]
    pub fn decoded_len(&self, row: usize) -> usize {
        let lo = self.codes_offsets[row] as usize;
        let hi = self.codes_offsets[row + 1] as usize;
        let mut total = 0;
        for &c in &self.codes[lo..hi] {
            let dlo = self.dict_offsets[c as usize] as usize;
            let dhi = self.dict_offsets[c as usize + 1] as usize;
            total += dhi - dlo;
        }
        total
    }

    /// Iterate the decoded bytes of `row` without materialising them, calling
    /// `f` on each contiguous dict slice. Returns early if `f` returns
    /// `false`. Useful for predicates that can short-circuit (e.g. `equals`,
    /// `starts_with`).
    #[inline]
    pub fn for_each_dict_slice<F: FnMut(&'a [u8]) -> bool>(&self, row: usize, mut f: F) -> bool {
        let lo = self.codes_offsets[row] as usize;
        let hi = self.codes_offsets[row + 1] as usize;
        for &c in &self.codes[lo..hi] {
            let dlo = self.dict_offsets[c as usize] as usize;
            let dhi = self.dict_offsets[c as usize + 1] as usize;
            if !f(&self.dict_bytes[dlo..dhi]) {
                return false;
            }
        }
        true
    }
}

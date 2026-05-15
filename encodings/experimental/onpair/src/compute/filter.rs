// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Filter that **shares the dictionary**. The previous implementation
//! decoded the whole array, filtered the canonical bytes, and re-trained
//! a brand-new OnPair dictionary on the surviving rows — order-of-
//! magnitude regressions on TPC-H Q22 at SF=10 traced back to that cost
//! (the customer table's `c_phone` column gets two consecutive filters,
//! each of which was paying full `Column::compress` training overhead).
//!
//! FSST-shape filter: keep `dict_bytes` + `dict_offsets` **identical**
//! to the input; rebuild only `codes`, `codes_offsets`,
//! `uncompressed_lengths`, and validity by walking the mask. No decode,
//! no retrain, no C++ call on the read path.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::OnPair;
use crate::OnPairArrayExt;

impl FilterKernel for OnPair {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let n_in = array.array().len();
        let n_out = mask.true_count();

        // Materialise the per-row offset arrays we walk during filtering.
        // The codes themselves we read through whatever ptype the
        // cascading compressor narrowed to — match_each_integer_ptype
        // dispatches on it below.
        let codes_offsets_arr = array
            .codes_offsets()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let codes_arr = array.codes().clone().execute::<PrimitiveArray>(ctx)?;

        let mut new_codes_offsets = BufferMut::<u32>::with_capacity(n_out + 1);

        // The cascading compressor may have narrowed `codes_offsets`
        // (e.g. u32 → u16 if every row's token count is small). Read
        // through whatever ptype it lives at — the values still fit in
        // `usize` when widened. Likewise for `codes`.
        let new_codes: ArrayRef = match_each_integer_ptype!(codes_offsets_arr.ptype(), |OP| {
            let codes_offsets = codes_offsets_arr.as_slice::<OP>();

            // First pass: sum the surviving token count so we reserve once.
            let mut new_codes_len: usize = 0;
            for r in 0..n_in {
                if mask.value(r) {
                    new_codes_len += (codes_offsets[r + 1] as usize) - (codes_offsets[r] as usize);
                }
            }

            // SAFETY: capacity reserved.
            unsafe { new_codes_offsets.push_unchecked(0u32) };

            match_each_integer_ptype!(codes_arr.ptype(), |P| {
                let codes = codes_arr.as_slice::<P>();
                let mut out = BufferMut::<P>::with_capacity(new_codes_len);
                let mut cursor: u32 = 0;
                for r in 0..n_in {
                    if mask.value(r) {
                        let lo = codes_offsets[r] as usize;
                        let hi = codes_offsets[r + 1] as usize;
                        // SAFETY: codes_offsets validated at construction.
                        let segment = unsafe { codes.get_unchecked(lo..hi) };
                        out.extend_from_slice(segment);
                        let segment_len = u32::try_from(hi - lo)
                            .map_err(|_| vortex_err!("token segment overflows u32"))?;
                        cursor = cursor
                            .checked_add(segment_len)
                            .ok_or_else(|| vortex_err!("codes_offsets overflow u32"))?;
                        // SAFETY: capacity reserved (n_out + 1 entries).
                        unsafe { new_codes_offsets.push_unchecked(cursor) };
                    }
                }
                out.freeze().into_array()
            })
        });

        // uncompressed_lengths + validity flow through the standard
        // primitive filter — these are short integer arrays so the cost
        // is negligible compared to the (avoided) recompress.
        let uncompressed_lengths = array.uncompressed_lengths().clone().filter(mask.clone())?;
        let validity = array.array_validity().filter(mask)?;

        Ok(Some(
            unsafe {
                OnPair::new_unchecked(
                    array.dtype().clone(),
                    array.dict_bytes_handle().clone(),
                    array.dict_offsets().clone(),
                    new_codes,
                    new_codes_offsets.freeze().into_array(),
                    uncompressed_lengths,
                    validity,
                    array.bits(),
                )
            }
            .into_array(),
        ))
    }
}

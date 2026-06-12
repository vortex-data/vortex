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
//! `uncompressed_lengths`, and validity. No decode, no retrain on the
//! read path.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::List;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::arrays::list::ListArrayExt;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::OnPair;
use crate::OnPairArrayExt;
use crate::OnPairArraySlotsExt;

impl FilterKernel for OnPair {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // OnPair's `codes` + `codes_offsets` are a list of token runs,
        // analogous to FSST's `codes` VarBin child. Delegate to the standard
        // List filter so sparse masks can filter the encoded child directly.
        let codes = unsafe {
            ListArray::new_unchecked(
                array.codes().clone(),
                array.codes_offsets().clone(),
                Validity::NonNullable,
            )
        };
        let filtered_codes_ref = <List as FilterKernel>::filter(codes.as_view(), mask, ctx)?
            .vortex_expect("List filter kernel always returns Some");
        let filtered_codes = filtered_codes_ref
            .try_downcast::<List>()
            .ok()
            .vortex_expect("must be List");

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
                    filtered_codes.elements().clone(),
                    filtered_codes.offsets().clone(),
                    uncompressed_lengths,
                    validity,
                    array.bits(),
                )
            }
            .into_array(),
        ))
    }
}

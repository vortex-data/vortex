// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Filter is implemented as a re-compress through canonical because OnPair's
//! `codes` for surviving rows would also need to be re-laid out (the codes
//! belong to whole rows, not single elements), and re-training keeps the
//! resulting dictionary tight to the surviving data. Slice is cheaper — see
//! `slice.rs` — because we can just sub-slice `codes_offsets` /
//! `uncompressed_lengths`.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::OnPair;
use crate::compress::DEFAULT_DICT12_CONFIG;
use crate::compress::onpair_compress_array;

impl FilterKernel for OnPair {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let canonical = array
            .array()
            .clone()
            .execute::<Canonical>(ctx)?
            .into_array();
        let filtered = canonical.filter(mask.clone())?;
        Ok(Some(
            onpair_compress_array(&filtered, DEFAULT_DICT12_CONFIG, ctx)?.into_array(),
        ))
    }
}

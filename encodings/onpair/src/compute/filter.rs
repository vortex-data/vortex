// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
        // OnPair does not currently expose a `take`-style compressed-domain
        // reshuffle, so we materialise to the canonical view, filter, and
        // recompress with the same training config. This preserves end-to-end
        // semantics; a future native filter kernel would skip the round-trip.
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

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::filter::FilterKernel;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::FSST;
use crate::FSSTArrayExt;

impl FilterKernel for FSST {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Directly invoke VarBin's FilterKernel on the codes child.
        let codes = array.codes();
        let codes = codes.as_view();
        let filtered_codes_ref = <VarBin as FilterKernel>::filter(codes, mask, ctx)?
            .vortex_expect("VarBin filter kernel always returns Some");
        let filtered_codes = filtered_codes_ref
            .try_downcast::<VarBin>()
            .ok()
            .vortex_expect("must be VarBin");

        Ok(Some(
            FSST::try_new(
                array.dtype().clone(),
                array.symbols().clone(),
                array.symbol_lengths().clone(),
                filtered_codes,
                array.uncompressed_lengths().filter(mask.clone())?,
            )?
            .into_array(),
        ))
    }
}

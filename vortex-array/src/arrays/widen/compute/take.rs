// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Widen;
use crate::arrays::dict::TakeExecute;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::wrap_values;

impl TakeExecute for Widen {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Execute the gather, so that no lazy `Dict` sits under the `Widen` (see the `widen`
        // module docs). `Columnar` keeps a constant gather constant.
        let values = array
            .values()
            .take(indices.clone())?
            .execute::<Columnar>(ctx)?
            .into_array();
        let dtype = array.dtype().with_nullability(values.dtype().nullability());

        wrap_values(values, dtype).map(Some)
    }
}

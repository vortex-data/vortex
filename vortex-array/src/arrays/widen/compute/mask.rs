// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::wrap_values;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::mask::MaskReduce;

impl MaskReduce for Widen {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let values = array.values().clone().mask(mask.clone())?;
        let dtype = array.dtype().with_nullability(values.dtype().nullability());

        wrap_values(values, dtype).map(Some)
    }
}

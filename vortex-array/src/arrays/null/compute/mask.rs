// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Null;
use crate::scalar_fn::fns::mask::MaskReduce;

impl MaskReduce for Null {
    fn mask(array: ArrayView<'_, Null>, _mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // Null array is already all nulls, masking has no effect.
        Ok(Some(array.array().clone()))
    }
}

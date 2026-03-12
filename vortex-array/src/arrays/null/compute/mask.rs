// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Null;
use crate::arrays::NullArray;
use crate::scalar_fn::fns::mask::MaskReduce;

impl MaskReduce for Null {
    fn mask(array: &NullArray, _mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // Null array is already all nulls, masking has no effect.
        Ok(Some(array.clone().into_array()))
    }
}

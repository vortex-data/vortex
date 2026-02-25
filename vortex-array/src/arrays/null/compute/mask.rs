// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::NullArray;
use crate::arrays::NullVTable;
use crate::scalar_fn::MaskReduce;

impl MaskReduce for NullVTable {
    fn mask(array: &NullArray, _mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // Null array is already all nulls, masking has no effect.
        Ok(Some(array.to_array()))
    }
}

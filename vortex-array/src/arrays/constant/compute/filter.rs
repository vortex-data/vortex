// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::filter::FilterReduce;

impl FilterReduce for ConstantVTable {
    fn filter(array: &ConstantArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ConstantArray::new(array.scalar().clone(), mask.true_count()).into_array(),
        ))
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::filter::FilterReduce;

impl FilterReduce for Constant {
    fn filter(array: ArrayView<'_, Constant>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ConstantArray::new(array.scalar().clone(), mask.true_count()).into_array(),
        ))
    }
}

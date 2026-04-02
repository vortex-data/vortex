// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Null;
use crate::arrays::NullArray;
use crate::arrays::filter::FilterReduce;

impl FilterReduce for Null {
    fn filter(_array: ArrayView<'_, Null>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(NullArray::new(mask.true_count()).into_array()))
    }
}

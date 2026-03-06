// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::filter::FilterReduce;
use crate::arrays::null::NullArray;
use crate::arrays::null::NullVTable;

impl FilterReduce for NullVTable {
    fn filter(_array: &NullArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(NullArray::new(mask.true_count()).into_array()))
    }
}

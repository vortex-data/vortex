// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Masked;
use crate::arrays::masked::MaskedData;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Masked {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let child = array.child().slice(range.clone())?;
        let validity = array.validity().slice(range)?;

        Ok(Some(MaskedData::try_new(child, validity)?.into_array()))
    }
}

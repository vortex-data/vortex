// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_error::VortexResult;

use crate::Binned;

pub(crate) static RULES: ParentRuleSet<Binned> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(Binned))]);

impl SliceReduce for Binned {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            crate::array::slice(array, range.start, range.end)?.into_array(),
        ))
    }
}

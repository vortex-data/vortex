// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::runend::RunEndArrayExt;
use vortex_array::arrays::runend::RunEndArraySlotsExt;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_error::VortexResult;

use crate::list_contains::ListContains;
use crate::list_contains::ListContainsElementReduce;
use crate::list_contains::ListContainsOptions;
use crate::list_contains::PreparedSet;

/// Pushes `list_contains` of a prepared set into the values of run-end encoded needles.
impl ListContainsElementReduce for RunEnd {
    fn list_contains(
        list: &ArrayRef,
        needles: ArrayView<'_, RunEnd>,
        options: &ListContainsOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(set) = PreparedSet::set_of(list) else {
            return Ok(None);
        };

        // The values get the shared set at their own length, which can exceed the rows of a
        // zero-length slice that kept its runs.
        let values = needles.values();
        let set = ScalarFnArray::try_new_with_len(PreparedSet.bind(set.clone()), [], values.len())?
            .into_array();
        let values = ListContains::try_new_opts(set, values.clone(), *options)?.into_array();

        // SAFETY: the ends are unchanged, and the values have one result for each run.
        Ok(Some(
            unsafe {
                RunEnd::new_unchecked(
                    needles.ends().clone(),
                    values,
                    needles.offset(),
                    needles.len(),
                )
            }
            .into_array(),
        ))
    }
}

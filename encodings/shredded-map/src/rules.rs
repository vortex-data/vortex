// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Slice, take and filter push down into every child, keeping the columns row-aligned.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::dict::TakeReduce;
use vortex_array::arrays::dict::TakeReduceAdaptor;
use vortex_array::arrays::filter::FilterReduce;
use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ShreddedMap;
use crate::array::ShreddedMapArrayExt;
use crate::array::ShreddedMapArraySlotsExt;
use crate::array::fields_struct;
use crate::array::make_parts;

pub(crate) static RULES: ParentRuleSet<ShreddedMap> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(ShreddedMap)),
    ParentRuleSet::lift(&SliceReduceAdaptor(ShreddedMap)),
    ParentRuleSet::lift(&TakeReduceAdaptor(ShreddedMap)),
]);

fn map_children(
    array: ArrayView<'_, ShreddedMap>,
    repeats: Option<ArrayRef>,
    f: impl Fn(&ArrayRef) -> VortexResult<ArrayRef>,
) -> VortexResult<ArrayRef> {
    let residual = f(array.residual())?;
    let columns = array
        .column_arrays()
        .iter()
        .map(&f)
        .collect::<VortexResult<Vec<_>>>()?;
    // Rebuild the struct from the selected fields so it stays a struct array.
    let fields = fields_struct(array.data().columns(), columns, residual.len())?;
    let parts = make_parts(
        residual.dtype().clone(),
        residual.len(),
        array.data().columns().into(),
        residual,
        repeats,
        fields,
    );
    // SAFETY: applying the same row selection to every child preserves the layout invariants.
    Ok(unsafe { vortex_array::Array::from_parts_unchecked(parts) }.into_array())
}

impl SliceReduce for ShreddedMap {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // A sliced row still equals its predecessor, except the new first row.
        let repeats = array
            .repeats()
            .map(|r| -> VortexResult<ArrayRef> {
                let sliced = r.slice(range.clone())?;
                if sliced.is_empty() {
                    return Ok(sliced);
                }
                let first = BoolArray::from_iter([false]).into_array();
                let rest = sliced.slice(1..sliced.len())?;
                Ok(ChunkedArray::try_new([first, rest], sliced.dtype().clone())?.into_array())
            })
            .transpose()?;
        map_children(array, repeats, |c| c.slice(range.clone())).map(Some)
    }
}

impl TakeReduce for ShreddedMap {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        map_children(array, None, |c| c.take(indices.clone())).map(Some)
    }
}

impl FilterReduce for ShreddedMap {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let indices = crate::flat::mask_indices(mask);
        map_children(array, None, |c| c.take(indices.clone())).map(Some)
    }
}

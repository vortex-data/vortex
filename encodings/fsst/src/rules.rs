// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::SliceVTable;
use vortex_array::arrays::VarBinVTable;
use vortex_array::matchers::Exact;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::FSSTArray;
use crate::FSSTVTable;

pub(super) const RULES: ParentRuleSet<FSSTVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&FSSTSliceRule)]);

/// A rule to push slice operations through FSST encoding.
///
/// Transforms: Slice(FSST(symbols, symbol_lengths, codes, uncompressed_lengths))
///          -> FSST(symbols, symbol_lengths, sliced_codes, sliced_uncompressed_lengths)
///
/// This works because FSST encoding is element-wise - each element has its own
/// compressed codes and uncompressed length. The symbol table is shared and
/// remains unchanged.
#[derive(Debug)]
struct FSSTSliceRule;

impl ArrayParentReduceRule<FSSTVTable> for FSSTSliceRule {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Exact<SliceVTable> {
        // SAFETY: SliceVTable is a valid VTable with a stable ID
        unsafe { Exact::new_unchecked(SliceVTable.id()) }
    }

    fn reduce_parent(
        &self,
        fsst: &FSSTArray,
        parent: &SliceArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let range = parent.slice_range().clone();

        // Slice the codes VarBinArray directly (not through SliceArray wrapper), since
        // FSSTArray requires the result to be a VarBin.
        let sliced_codes = VarBinVTable::slice(fsst.codes(), range.clone())?
            .ok_or_else(|| vortex_err!("VarBinVTable::slice returned None"))?
            .as_::<VarBinVTable>()
            .clone();

        let sliced_uncompressed_lengths = fsst.uncompressed_lengths().slice(range);

        // SAFETY: slicing the codes leaves the symbol table intact and valid
        Ok(Some(
            unsafe {
                FSSTArray::new_unchecked(
                    fsst.dtype().clone(),
                    fsst.symbols().clone(),
                    fsst.symbol_lengths().clone(),
                    sliced_codes,
                    sliced_uncompressed_lengths,
                )
            }
            .into_array(),
        ))
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::compute::NotReduce;
use vortex_array::compute::NotReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_error::VortexResult;

use crate::SparseArray;
use crate::SparseVTable;

pub(super) const RULES: ParentRuleSet<SparseVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&NotReduceAdaptor(SparseVTable))]);

impl NotReduce for SparseVTable {
    fn invert(array: &SparseArray) -> VortexResult<Option<ArrayRef>> {
        let inverted_fill = array.fill_scalar().as_bool().invert().into_scalar();
        let inverted_patches = array.patches().clone().map_values(|values| values.not())?;
        Ok(Some(
            SparseArray::try_new_from_patches(inverted_patches, inverted_fill)?.into_array(),
        ))
    }
}

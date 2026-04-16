// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Variant;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::arrays::variant::VariantArrayExt;
use crate::arrays::variant::rebuild_variant_array;
use crate::optimizer::rules::ParentRuleSet;

pub(crate) const PARENT_RULES: ParentRuleSet<Variant> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(Variant))]);

impl SliceReduce for Variant {
    fn slice(
        array: ArrayView<'_, Self>,
        range: std::ops::Range<usize>,
    ) -> VortexResult<Option<ArrayRef>> {
        let core_storage = array.core_storage().slice(range.clone())?;
        let rebuilt = rebuild_variant_array(&array, core_storage, || {
            array.shredded().map(|child| child.slice(range)).transpose()
        })?;

        Ok(Some(rebuilt.into_array()))
    }
}

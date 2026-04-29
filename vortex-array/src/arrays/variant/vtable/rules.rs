// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Masked;
use crate::arrays::MaskedArray;
use crate::arrays::Variant;
use crate::arrays::dict::TakeReduce;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduce;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::masked::MaskedArrayExt;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::arrays::variant::VariantArrayExt;
use crate::arrays::variant::rebuild_variant_array;
use crate::builtins::ArrayBuiltins;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::validity::Validity;

pub(crate) const PARENT_RULES: ParentRuleSet<Variant> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(Variant)),
    ParentRuleSet::lift(&VariantMaskedReduceRule),
    ParentRuleSet::lift(&SliceReduceAdaptor(Variant)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Variant)),
]);

impl FilterReduce for Variant {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let core_storage = array.core_storage().filter(mask.clone())?;
        let rebuilt = rebuild_variant_array(&array, core_storage, || {
            array
                .shredded()
                .map(|child| child.filter(mask.clone()))
                .transpose()
        })?;

        Ok(Some(rebuilt.into_array()))
    }
}

#[derive(Default, Debug)]
struct VariantMaskedReduceRule;

impl ArrayParentReduceRule<Variant> for VariantMaskedReduceRule {
    type Parent = Masked;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Variant>,
        parent: ArrayView<'_, Masked>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }

        let validity = parent.masked_validity();
        if matches!(validity, Validity::AllValid) && parent.dtype() == array.dtype() {
            return Ok(Some(array.array().clone()));
        }

        let core_storage =
            MaskedArray::try_new(array.core_storage().clone(), validity.clone())?.into_array();
        let rebuilt = rebuild_variant_array(&array, core_storage, || {
            array
                .shredded()
                .map(|child| child.mask(validity.to_array(array.len())))
                .transpose()
        })?;

        Ok(Some(rebuilt.into_array()))
    }
}

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

impl TakeReduce for Variant {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let core_storage = array.core_storage().take(indices.clone())?;
        let rebuilt = rebuild_variant_array(&array, core_storage, || {
            array
                .shredded()
                .map(|child| child.take(indices.clone()))
                .transpose()
        })?;

        Ok(Some(rebuilt.into_array()))
    }
}

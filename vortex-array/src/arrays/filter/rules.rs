// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Filter;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::filter::FilterArraySlotsExt;
use crate::arrays::filter::FilterReduce;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::filter::execute::buffer::prepare_mask_for_reuse;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::struct_::StructDataParts;
use crate::builtins::ArrayBuiltins;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::optimizer::rules::ReduceRuleSet;
use crate::scalar_fn::fns::get_item::GetItem;

pub(super) const PARENT_RULES: ParentRuleSet<Filter> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(Filter)),
    ParentRuleSet::lift(&FilterGetItemRule),
]);

pub(super) const RULES: ReduceRuleSet<Filter> =
    ReduceRuleSet::new(&[&TrivialFilterRule, &FilterStructRule]);

impl FilterReduce for Filter {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let combined_mask = array.mask.intersect_by_rank(mask);
        let new_array = array.child().filter(combined_mask)?;

        Ok(Some(new_array))
    }
}

#[derive(Debug)]
struct FilterGetItemRule;

impl ArrayParentReduceRule<Filter> for FilterGetItemRule {
    type Parent = ExactScalarFn<GetItem>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Filter>,
        parent: ScalarFnArrayView<'_, GetItem>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let field = array.child().get_item(parent.options.clone())?;
        Ok(Some(field.filter(array.filter_mask().clone())?))
    }
}

#[derive(Debug)]
struct TrivialFilterRule;

impl ArrayReduceRule<Filter> for TrivialFilterRule {
    fn reduce(&self, array: ArrayView<'_, Filter>) -> VortexResult<Option<ArrayRef>> {
        match array.filter_mask() {
            Mask::AllTrue(_) => Ok(Some(array.child().clone())),
            Mask::AllFalse(_) => Ok(Some(Canonical::empty(array.dtype()).into_array())),
            Mask::Values(_) => Ok(None),
        }
    }
}

/// A reduce rule that pushes a filter down into the fields of a StructArray.
#[derive(Debug)]
struct FilterStructRule;

impl ArrayReduceRule<Filter> for FilterStructRule {
    fn reduce(&self, array: ArrayView<'_, Filter>) -> VortexResult<Option<ArrayRef>> {
        let mask = array.filter_mask();
        let Some(struct_array) = array.child().as_opt::<Struct>() else {
            return Ok(None);
        };

        let len = mask.true_count();
        let StructDataParts {
            fields,
            struct_fields,
            validity,
            ..
        } = struct_array.into_owned().into_data_parts();

        if let Some(values) = mask.values() {
            let consumers = fields.len();
            prepare_mask_for_reuse(values, consumers);
        }

        let filtered_validity = validity.filter(mask)?;

        let filtered_fields = fields
            .iter()
            .map(|field| field.filter(mask.clone()))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Some(
            StructArray::new(
                struct_fields.names().clone(),
                filtered_fields,
                len,
                filtered_validity,
            )
            .into_array(),
        ))
    }
}

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
use crate::arrays::filter::FilterArrayExt;
use crate::arrays::filter::FilterReduce;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::struct_::StructDataParts;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::optimizer::rules::ReduceRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<Filter> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&FilterReduceAdaptor(Filter))]);

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

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
use crate::arrays::struct_::StructArrayParts;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::optimizer::rules::ReduceRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<Filter> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&FilterFilterRule)]);

pub(super) const RULES: ReduceRuleSet<Filter> =
    ReduceRuleSet::new(&[&TrivialFilterRule, &FilterStructRule]);

/// A simple redecution rule that simplifies a [`FilterArray`] whose child is also a
/// [`FilterArray`].
#[derive(Debug)]
struct FilterFilterRule;

impl ArrayParentReduceRule<Filter> for FilterFilterRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, Filter>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let combined_mask = child.mask.intersect_by_rank(&parent.mask);
        let new_array = child.child().filter(combined_mask)?;

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
        let StructArrayParts {
            fields,
            struct_fields,
            validity,
            ..
        } = struct_array.into_owned().into_data().into_parts();

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

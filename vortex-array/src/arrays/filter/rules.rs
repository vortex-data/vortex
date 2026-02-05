// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::arrays::StructArray;
use crate::arrays::StructArrayParts;
use crate::arrays::StructVTable;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::optimizer::rules::ReduceRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<FilterVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&FilterFilterRule)]);

pub(super) const RULES: ReduceRuleSet<FilterVTable> = ReduceRuleSet::new(&[&FilterStructRule]);

/// A simple redecution rule that simplifies a [`FilterArray`] whose child is also a
/// [`FilterArray`].
#[derive(Debug)]
struct FilterFilterRule;

impl ArrayParentReduceRule<FilterVTable> for FilterFilterRule {
    type Parent = FilterVTable;

    fn reduce_parent(
        &self,
        child: &FilterArray,
        parent: &FilterArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let combined_mask = child.mask.intersect_by_rank(&parent.mask);
        let new_array = child.child.filter(combined_mask)?;

        Ok(Some(new_array.into_array()))
    }
}

/// A reduce rule that pushes a filter down into the fields of a StructArray.
#[derive(Debug)]
struct FilterStructRule;

impl ArrayReduceRule<FilterVTable> for FilterStructRule {
    fn reduce(&self, array: &FilterArray) -> VortexResult<Option<ArrayRef>> {
        let mask = array.filter_mask();
        let Some(struct_array) = array.child().as_opt::<StructVTable>() else {
            return Ok(None);
        };

        let len = mask.true_count();
        let StructArrayParts {
            fields,
            struct_fields,
            validity,
            ..
        } = struct_array.clone().into_parts();

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

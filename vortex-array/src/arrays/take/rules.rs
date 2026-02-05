// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::StructArray;
use crate::arrays::StructArrayParts;
use crate::arrays::StructVTable;
use crate::arrays::TakeArray;
use crate::arrays::TakeVTable;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::optimizer::rules::ReduceRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<TakeVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&TakeTakeRule)]);

pub(super) const RULES: ReduceRuleSet<TakeVTable> = ReduceRuleSet::new(&[&TakeStructRule]);

/// A simple reduction rule that simplifies a [`TakeArray`] whose child is also a
/// [`TakeArray`].
#[derive(Debug)]
struct TakeTakeRule;

impl ArrayParentReduceRule<TakeVTable> for TakeTakeRule {
    type Parent = TakeVTable;

    fn reduce_parent(
        &self,
        child: &TakeArray,
        parent: &TakeArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Take(Take(arr, indices1), indices2) = Take(arr, Take(indices1, indices2))
        // We need to take from the inner indices using the outer indices
        let new_indices = child.indices.take(parent.indices.clone())?;
        let new_array = child.child.take(new_indices)?;

        Ok(Some(new_array.into_array()))
    }
}

/// A reduce rule that pushes a take down into the fields of a StructArray.
#[derive(Debug)]
struct TakeStructRule;

impl ArrayReduceRule<TakeVTable> for TakeStructRule {
    fn reduce(&self, array: &TakeArray) -> VortexResult<Option<ArrayRef>> {
        let indices = array.indices();
        let Some(struct_array) = array.child().as_opt::<StructVTable>() else {
            return Ok(None);
        };

        let len = indices.len();
        let StructArrayParts {
            fields,
            struct_fields,
            validity,
            ..
        } = struct_array.clone().into_parts();

        let taken_validity = validity.take(indices)?;

        let taken_fields = fields
            .iter()
            .map(|field| field.take(indices.clone()))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Some(
            StructArray::new(
                struct_fields.names().clone(),
                taken_fields,
                len,
                taken_validity,
            )
            .into_array(),
        ))
    }
}

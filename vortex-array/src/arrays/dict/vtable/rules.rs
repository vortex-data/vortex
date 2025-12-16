// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayEq;
use crate::ArrayRef;
use crate::IntoArray;
use crate::Precision;
use crate::arrays::AnyScalarFn;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::DictArray;
use crate::arrays::DictVTable;
use crate::arrays::ScalarFnArray;
use crate::optimizer::ArrayOptimizer;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<DictVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&DictionaryScalarFnValuesPushDownRule),
    ParentRuleSet::lift(&DictionaryScalarFnCodesPullUpRule),
]);

/// Push down a scalar function to run only over the values of a dictionary array.
#[derive(Debug)]
struct DictionaryScalarFnValuesPushDownRule;

impl ArrayParentReduceRule<DictVTable> for DictionaryScalarFnValuesPushDownRule {
    type Parent = AnyScalarFn;

    fn parent(&self) -> Self::Parent {
        AnyScalarFn
    }

    fn reduce_parent(
        &self,
        array: &DictArray,
        parent: &ScalarFnArray,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Check that the scalar function can actually be pushed down.
        let sig = parent.scalar_fn().signature();

        // If the scalar function is fallible, we cannot push it down since it may fail over a
        // value that isn't referenced by any code.
        if !array.all_values_referenced && sig.is_fallible() {
            tracing::trace!(
                "Not pushing down fallible scalar function {} over dictionary with sparse codes {}",
                parent.scalar_fn(),
                array.display_tree(),
            );
            return Ok(None);
        }

        // Check that all siblings are constant
        // TODO(ngates): we can also support other dictionaries if the values are the same!
        if !parent
            .children()
            .iter()
            .enumerate()
            .all(|(idx, c)| idx == child_idx || c.is::<ConstantVTable>())
        {
            return Ok(None);
        }

        // If the scalar function is null-sensitive, then we cannot push it down to values if
        // we have any nulls in the codes.
        if array.codes.dtype().is_nullable() && !array.codes.all_valid() && sig.is_null_sensitive()
        {
            tracing::trace!(
                "Not pushing down null-sensitive scalar function {} over dictionary with null codes {}",
                parent.scalar_fn(),
                array.display_tree(),
            );
            return Ok(None);
        }

        // Now we push the parent scalar function into the dictionary values.
        let values_len = array.values().len();
        let mut new_children = Vec::with_capacity(parent.children().len());
        for (idx, child) in parent.children().iter().enumerate() {
            if idx == child_idx {
                new_children.push(array.values().clone());
            } else {
                let scalar = child.as_::<ConstantVTable>().scalar().clone();
                new_children.push(ConstantArray::new(scalar, values_len).into_array());
            }
        }

        let new_values =
            ScalarFnArray::try_new(parent.scalar_fn().clone(), new_children, values_len)?
                .into_array()
                .optimize()?;

        let new_dict =
            unsafe { DictArray::new_unchecked(array.codes().clone(), new_values) }.into_array();

        Ok(Some(new_dict))
    }
}

#[derive(Debug)]
struct DictionaryScalarFnCodesPullUpRule;

impl ArrayParentReduceRule<DictVTable> for DictionaryScalarFnCodesPullUpRule {
    type Parent = AnyScalarFn;

    fn parent(&self) -> Self::Parent {
        AnyScalarFn
    }

    fn reduce_parent(
        &self,
        array: &DictArray,
        parent: &ScalarFnArray,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Check that all siblings are dictionaries with the same codes as us.
        if !parent.children().iter().enumerate().all(|(idx, c)| {
            idx == child_idx
                || c.as_opt::<DictVTable>().is_some_and(|c| {
                    c.values().len() == array.values().len()
                        && c.codes().array_eq(array.codes(), Precision::Value)
                })
        }) {
            return Ok(None);
        }

        let mut new_children = Vec::with_capacity(parent.children().len());
        for (idx, child) in parent.children().iter().enumerate() {
            if idx == child_idx {
                new_children.push(array.values().clone());
            } else {
                new_children.push(child.as_::<DictVTable>().values().clone());
            }
        }

        let new_values =
            ScalarFnArray::try_new(parent.scalar_fn().clone(), new_children, array.values.len())?
                .into_array()
                .optimize()?;

        let new_dict =
            unsafe { DictArray::new_unchecked(array.codes().clone(), new_values) }.into_array();

        Ok(Some(new_dict))
    }
}

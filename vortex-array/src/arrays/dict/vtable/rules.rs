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
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::arrays::ScalarFnArray;
use crate::builtins::ArrayBuiltins;
use crate::expr::Pack;
use crate::optimizer::ArrayOptimizer;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<DictVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&DictionaryFilterPushDownRule),
    ParentRuleSet::lift(&DictionaryScalarFnValuesPushDownRule),
    ParentRuleSet::lift(&DictionaryScalarFnCodesPullUpRule),
]);

#[derive(Debug)]
struct DictionaryFilterPushDownRule;

impl ArrayParentReduceRule<DictVTable> for DictionaryFilterPushDownRule {
    type Parent = FilterVTable;

    fn reduce_parent(
        &self,
        array: &DictArray,
        parent: &FilterArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let new_codes = array.codes().filter(parent.filter_mask().clone())?;
        let new_dict =
            unsafe { DictArray::new_unchecked(new_codes, array.values().clone()) }.into_array();
        Ok(Some(new_dict))
    }
}

/// Push down a scalar function to run only over the values of a dictionary array.
#[derive(Debug)]
struct DictionaryScalarFnValuesPushDownRule;

impl ArrayParentReduceRule<DictVTable> for DictionaryScalarFnValuesPushDownRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        array: &DictArray,
        parent: &ScalarFnArray,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Check that the scalar function can actually be pushed down.
        let sig = parent.scalar_fn().signature();

        // Don't push down pack expressions since we might want to unpack them in exporters
        // later.
        if parent.scalar_fn().is::<Pack>() {
            return Ok(None);
        }

        // If the dictionary has less codes than values don't push down this might
        // happen if the dictionary is sliced.
        if array.values().len() > array.codes().len() {
            return Ok(None);
        }

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
        if array.codes.dtype().is_nullable() && !array.codes.all_valid()? && sig.is_null_sensitive()
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

        // We can only push down null-sensitive functions when we have all-valid codes.
        // In these cases, we cannot have the codes influence the nullability of the output DType.
        // Therefore, we cast the codes to be non-nullable and then cast the dictionary output
        // back to nullable if needed.
        if sig.is_null_sensitive() && array.codes().dtype().is_nullable() {
            let new_codes = array.codes().cast(array.codes().dtype().as_nonnullable())?;
            let new_dict = unsafe { DictArray::new_unchecked(new_codes, new_values) }.into_array();
            return Ok(Some(new_dict.cast(parent.dtype().clone())?));
        }

        Ok(Some(
            unsafe { DictArray::new_unchecked(array.codes().clone(), new_values) }.into_array(),
        ))
    }
}

#[derive(Debug)]
struct DictionaryScalarFnCodesPullUpRule;

impl ArrayParentReduceRule<DictVTable> for DictionaryScalarFnCodesPullUpRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        array: &DictArray,
        parent: &ScalarFnArray,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Don't attempt to pull up if there are less than 2 siblings.
        if parent.children().len() < 2 {
            return Ok(None);
        }

        // Check that all siblings are dictionaries, and have the same number of values as us.
        // This is a cheap first loop.
        if !parent.children().iter().enumerate().all(|(idx, c)| {
            idx == child_idx
                || c.as_opt::<DictVTable>()
                    .is_some_and(|c| c.values().len() == array.values().len())
        }) {
            return Ok(None);
        }

        // Now run the slightly more expensive check that all siblings have the same codes as us.
        // We use the cheaper Precision::Ptr to avoid doing data comparisons.
        if !parent.children().iter().enumerate().all(|(idx, c)| {
            idx == child_idx
                || c.as_opt::<DictVTable>()
                    .is_some_and(|c| c.codes().array_eq(array.codes(), Precision::Value))
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

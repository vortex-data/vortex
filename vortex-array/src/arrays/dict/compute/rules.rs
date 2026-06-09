// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayEq;
use crate::ArrayRef;
use crate::EqMode;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::ScalarFn;
use crate::arrays::ScalarFnArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::scalar_fn::AnyScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::builtins::ArrayBuiltins;
use crate::optimizer::ArrayOptimizer;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::like::LikeReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;
use crate::scalar_fn::fns::pack::Pack;
use crate::validity::Validity;

pub(crate) const PARENT_RULES: ParentRuleSet<Dict> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(Dict)),
    ParentRuleSet::lift(&CastReduceAdaptor(Dict)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Dict)),
    ParentRuleSet::lift(&LikeReduceAdaptor(Dict)),
    ParentRuleSet::lift(&DictionaryScalarFnValuesPushDownRule),
    ParentRuleSet::lift(&DictionaryScalarFnCodesPullUpRule),
    ParentRuleSet::lift(&SliceReduceAdaptor(Dict)),
]);

/// Push down a scalar function to run only over the values of a dictionary array.
#[derive(Debug)]
struct DictionaryScalarFnValuesPushDownRule;

impl ArrayParentReduceRule<Dict> for DictionaryScalarFnValuesPushDownRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Dict>,
        parent: ArrayView<'_, ScalarFn>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Check that the scalar function can actually be pushed down.
        let sig = parent.scalar_fn().signature();

        // Don't push down pack expressions since we might want to unpack them in exporters
        // later.
        if parent.scalar_fn().is::<Pack>() {
            return Ok(None);
        }

        // Don't push down cast operations — CastReduceAdaptor handles these eagerly.
        // If it declined (returned None), we must fall through to the canonical path
        // rather than creating a lazy cast inside the dictionary values.
        if parent.scalar_fn().is::<Cast>() {
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
                Dict.id(),
            );
            return Ok(None);
        }

        // Check that all siblings are constant
        // TODO(ngates): we can also support other dictionaries if the values are the same!
        if !parent
            .iter_children()
            .enumerate()
            .all(|(idx, c)| idx == child_idx || c.is::<Constant>())
        {
            return Ok(None);
        }

        // If the scalar function is null-sensitive, then we cannot push it down to values if
        // we have any nulls in the codes.
        if array.codes().dtype().is_nullable()
            && !matches!(
                array.codes().validity()?,
                Validity::NonNullable | Validity::AllValid
            )
            && sig.is_null_sensitive()
        {
            tracing::trace!(
                "Not pushing down null-sensitive scalar function {} over dictionary with null codes {}",
                parent.scalar_fn(),
                Dict.id(),
            );
            return Ok(None);
        }

        // Now we push the parent scalar function into the dictionary values.
        let values_len = array.values().len();
        let mut new_children = Vec::with_capacity(parent.nchildren());
        for (idx, child) in parent.iter_children().enumerate() {
            if idx == child_idx {
                new_children.push(array.values().clone());
            } else {
                let scalar = child.as_::<Constant>().scalar().clone();
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

impl ArrayParentReduceRule<Dict> for DictionaryScalarFnCodesPullUpRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Dict>,
        parent: ArrayView<'_, ScalarFn>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Don't attempt to pull up if there are less than 2 siblings.
        if parent.nchildren() < 2 {
            return Ok(None);
        }

        // Check that all siblings are dictionaries, and have the same number of values as us.
        // This is a cheap first loop.
        if !parent.iter_children().enumerate().all(|(idx, c)| {
            idx == child_idx
                || c.as_opt::<Dict>()
                    .is_some_and(|c| c.values().len() == array.values().len())
        }) {
            return Ok(None);
        }

        // Now run the slightly more expensive check that all siblings have the same codes as us.
        if !parent.iter_children().enumerate().all(|(idx, c)| {
            idx == child_idx
                || c.as_opt::<Dict>()
                    .is_some_and(|c| c.codes().array_eq(array.codes(), EqMode::Value))
        }) {
            return Ok(None);
        }

        let mut new_children = Vec::with_capacity(parent.nchildren());
        for (idx, child) in parent.iter_children().enumerate() {
            if idx == child_idx {
                new_children.push(array.values().clone());
            } else {
                new_children.push(child.as_::<Dict>().values().clone());
            }
        }

        let new_values = ScalarFnArray::try_new(
            parent.scalar_fn().clone(),
            new_children,
            array.values().len(),
        )?
        .into_array()
        .optimize()?;

        let new_dict =
            unsafe { DictArray::new_unchecked(array.codes().clone(), new_values) }.into_array();

        Ok(Some(new_dict))
    }
}

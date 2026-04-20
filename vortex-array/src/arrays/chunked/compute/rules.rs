// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::ScalarFnVTable;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::scalar_fn::AnyScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::optimizer::ArrayOptimizer;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleDense;
use crate::optimizer::rules::ParentRuleEntry;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullReduceAdaptor;

static KEYED_PARENT_RULES: [ParentRuleEntry<Chunked>; 2] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(Chunked)),
    ParentRuleSet::lift_id(
        CachedId::new("vortex.fill_null"),
        &FillNullReduceAdaptor(Chunked),
    ),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<Chunked> = ParentRuleDense::new();

pub(crate) static PARENT_RULES: ParentRuleSet<Chunked> = ParentRuleSet::new_indexed(
    &KEYED_PARENT_RULES,
    &KEYED_PARENT_RULES_DENSE,
    &[
        ParentRuleSet::lift(&ChunkedUnaryScalarFnPushDownRule),
        ParentRuleSet::lift(&ChunkedConstantScalarFnPushDownRule),
    ],
);

/// Push down any unary scalar function through chunked arrays.
#[derive(Debug)]
struct ChunkedUnaryScalarFnPushDownRule;
impl ArrayParentReduceRule<Chunked> for ChunkedUnaryScalarFnPushDownRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Chunked>,
        parent: ArrayView<'_, ScalarFnVTable>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if parent.nchildren() != 1 {
            return Ok(None);
        }

        let new_chunks: Vec<_> = array
            .iter_chunks()
            .map(|chunk| {
                ScalarFnArray::try_new(
                    parent.scalar_fn().clone(),
                    vec![chunk.clone()],
                    chunk.len(),
                )?
                .into_array()
                .optimize()
            })
            .try_collect()?;

        Ok(Some(
            unsafe { ChunkedArray::new_unchecked(new_chunks, parent.dtype().clone()) }.into_array(),
        ))
    }
}

/// Push down non-unary scalar functions through chunked arrays where other siblings are constant.
#[derive(Debug)]
struct ChunkedConstantScalarFnPushDownRule;
impl ArrayParentReduceRule<Chunked> for ChunkedConstantScalarFnPushDownRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Chunked>,
        parent: ArrayView<'_, ScalarFnVTable>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        for (idx, child) in parent.iter_children().enumerate() {
            if idx == child_idx {
                continue;
            }
            if !child.is::<Constant>() {
                return Ok(None);
            }
        }

        let new_chunks: Vec<_> = array
            .iter_chunks()
            .map(|chunk| {
                let new_children: Vec<_> = parent
                    .iter_children()
                    .enumerate()
                    .map(|(idx, child)| {
                        if idx == child_idx {
                            chunk.clone()
                        } else {
                            ConstantArray::new(
                                child.as_::<Constant>().scalar().clone(),
                                chunk.len(),
                            )
                            .into_array()
                        }
                    })
                    .collect();

                ScalarFnArray::try_new(parent.scalar_fn().clone(), new_children, chunk.len())?
                    .into_array()
                    .optimize()
            })
            .try_collect()?;

        Ok(Some(
            unsafe { ChunkedArray::new_unchecked(new_chunks, parent.dtype().clone()) }.into_array(),
        ))
    }
}

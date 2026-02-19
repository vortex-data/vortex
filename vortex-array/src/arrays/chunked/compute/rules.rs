// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::AnyScalarFn;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::ScalarFnArray;
use crate::compute::CastReduceAdaptor;
use crate::expr::FillNullReduceAdaptor;
use crate::expr::ZipReduceAdaptor;
use crate::optimizer::ArrayOptimizer;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;

pub(crate) const PARENT_RULES: ParentRuleSet<ChunkedVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(ChunkedVTable)),
    ParentRuleSet::lift(&ChunkedUnaryScalarFnPushDownRule),
    ParentRuleSet::lift(&ChunkedConstantScalarFnPushDownRule),
    ParentRuleSet::lift(&FillNullReduceAdaptor(ChunkedVTable)),
    ParentRuleSet::lift(&ZipReduceAdaptor(ChunkedVTable)),
]);

/// Push down any unary scalar function through chunked arrays.
#[derive(Debug)]
struct ChunkedUnaryScalarFnPushDownRule;
impl ArrayParentReduceRule<ChunkedVTable> for ChunkedUnaryScalarFnPushDownRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        array: &ChunkedArray,
        parent: &ScalarFnArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if parent.children().len() != 1 {
            return Ok(None);
        }

        let new_chunks: Vec<_> = array
            .chunks
            .iter()
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
impl ArrayParentReduceRule<ChunkedVTable> for ChunkedConstantScalarFnPushDownRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        array: &ChunkedArray,
        parent: &ScalarFnArray,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        for (idx, child) in parent.children().iter().enumerate() {
            if idx == child_idx {
                continue;
            }
            if !child.is::<ConstantVTable>() {
                return Ok(None);
            }
        }

        let new_chunks: Vec<_> = array
            .chunks
            .iter()
            .map(|chunk| {
                let new_children: Vec<_> = parent
                    .children()
                    .iter()
                    .enumerate()
                    .map(|(idx, child)| {
                        if idx == child_idx {
                            chunk.clone()
                        } else {
                            ConstantArray::new(
                                child.as_::<ConstantVTable>().scalar().clone(),
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

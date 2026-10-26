// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::Constant;
use crate::arrays::ScalarFn;
use crate::arrays::ScalarFnArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::scalar_fn::AnyScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::builtin::cast::CastReduceAdaptor;
use crate::scalar_fn::builtin::fill_null::FillNullReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<Chunked> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(Chunked)),
    ParentRuleSet::lift(&ChunkedUnaryScalarFnPushDownRule),
    ParentRuleSet::lift(&ChunkedConstantScalarFnPushDownRule),
    ParentRuleSet::lift(&FillNullReduceAdaptor(Chunked)),
]);

/// Push down any unary scalar function through chunked arrays.
#[derive(Debug)]
struct ChunkedUnaryScalarFnPushDownRule;
impl ArrayParentReduceRule<Chunked> for ChunkedUnaryScalarFnPushDownRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Chunked>,
        parent: ArrayView<'_, ScalarFn>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if parent.nchildren() != 1 {
            return Ok(None);
        }

        let new_chunks: Vec<_> = array
            .iter_chunks()
            .map(|chunk| -> VortexResult<ArrayRef> {
                Ok(
                    ScalarFnArray::try_new(parent.scalar_fn().clone(), vec![chunk.clone()])?
                        .into_array(),
                )
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
        parent: ArrayView<'_, ScalarFn>,
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

        let mut offset = 0;
        let new_chunks: Vec<_> = array
            .iter_chunks()
            .map(|chunk| -> VortexResult<ArrayRef> {
                let range = offset..offset + chunk.len();
                offset += chunk.len();
                // Each sibling is sliced to the chunk's rows, so that every chunk shares what the
                // sibling shares rather than getting its own copy of the constant.
                let new_children: Vec<_> = parent
                    .iter_children()
                    .enumerate()
                    .map(|(idx, child)| {
                        if idx == child_idx {
                            Ok(chunk.clone())
                        } else {
                            child.slice(range.clone())
                        }
                    })
                    .try_collect()?;

                Ok(ScalarFnArray::try_new(parent.scalar_fn().clone(), new_children)?.into_array())
            })
            .try_collect()?;

        Ok(Some(
            unsafe { ChunkedArray::new_unchecked(new_chunks, parent.dtype().clone()) }.into_array(),
        ))
    }
}

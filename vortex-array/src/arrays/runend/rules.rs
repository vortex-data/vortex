// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArrayExt;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::arrays::scalar_fn::AnyScalarFn;
use crate::arrays::scalar_fn::ScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::dtype::DType;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullReduceAdaptor;

pub(super) const RULES: ParentRuleSet<RunEnd> = ParentRuleSet::new(&[
    // CastReduceAdaptor must come before RunEndScalarFnRule so that cast operations are executed
    // eagerly (surfacing out-of-range errors immediately) rather than being pushed lazily into
    // the values array by the generic scalar function push-down rule.
    ParentRuleSet::lift(&CastReduceAdaptor(RunEnd)),
    ParentRuleSet::lift(&RunEndScalarFnRule),
    ParentRuleSet::lift(&FillNullReduceAdaptor(RunEnd)),
]);

/// A rule to push down scalar functions through run-end encoding into the values array.
///
/// This only works if all other children of the scalar function array are constants.
#[derive(Debug)]
pub(crate) struct RunEndScalarFnRule;

impl ArrayParentReduceRule<RunEnd> for RunEndScalarFnRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        run_end: ArrayView<'_, RunEnd>,
        parent: ArrayView<'_, ScalarFn>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        for (idx, child) in parent.iter_children().enumerate() {
            if idx == child_idx {
                // Skip ourselves
                continue;
            }

            if !child.is::<Constant>() {
                // We can only push down if all other children are constants
                return Ok(None);
            }
        }

        // TODO(ngates): relax this constraint and implement run-end decoding for all vector types.
        if !matches!(parent.dtype(), DType::Bool(_) | DType::Primitive(..)) {
            return Ok(None);
        }

        let values_len = run_end.values().len();
        let mut new_children: Vec<ArrayRef> = parent.children().to_vec();
        for (idx, child) in new_children.iter_mut().enumerate() {
            if idx == child_idx {
                // Replace ourselves with run end values
                *child = run_end.values().clone();
                continue;
            }

            // Replace other children with their constant scalar value with length adjusted
            // to the length of the run end values.
            let constant = child.as_::<Constant>();
            *child = ConstantArray::new(constant.scalar().clone(), values_len).into_array();
        }

        let new_values =
            ScalarFnArray::try_new(parent.scalar_fn().clone(), new_children)?.into_array();

        Ok(Some(
            unsafe {
                RunEnd::new_unchecked(
                    run_end.ends().clone(),
                    new_values,
                    run_end.offset(),
                    run_end.len(),
                )
            }
            .into_array(),
        ))
    }
}

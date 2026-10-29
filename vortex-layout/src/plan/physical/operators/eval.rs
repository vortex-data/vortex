// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Applying an expression to each batch.

use vortex_array::expr::BoundExpression;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_error::VortexResult;

use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Step;

/// Applies an expression to each batch.
pub(crate) struct EvalStage {
    expression: BoundExpression,
}

impl EvalStage {
    pub(crate) fn new(expression: BoundExpression) -> Self {
        Self { expression }
    }
}

impl Operator for EvalStage {
    fn compute(&mut self, input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let Input::Batch(batch) = input else {
            return Ok(Step::End);
        };
        // `apply_bound` builds the expression lazily. Optimizing it here hands consumers the
        // reduced array, such as a struct for a pack of fields, rather than the expression node.
        let applied = batch.apply_bound(&self.expression)?;
        Ok(Step::Final(applied.optimize(cx.session())?))
    }
}

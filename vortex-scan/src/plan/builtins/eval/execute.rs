// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Applying an expression to each batch.

use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;

use crate::execution::operator::Cx;
use crate::execution::operator::Input;
use crate::execution::operator::Operator;
use crate::execution::operator::Step;

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
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        let Input::Batch(batch) = input else {
            return Ok(Step::End);
        };
        Ok(Step::Final(batch.apply_bound(&self.expression)?))
    }
}

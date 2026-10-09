// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Applying an expression to each batch.

use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;

use crate::plan::pipeline::Cx;
use crate::plan::pipeline::Input;
use crate::plan::pipeline::Operator;
use crate::plan::pipeline::Step;

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
        match input {
            Input::Chunk(batch) => Ok(Step::Last(batch.apply_bound(&self.expression)?)),
            Input::End => Ok(Step::Finished),
            Input::None => Ok(Step::Consumed),
        }
    }
}

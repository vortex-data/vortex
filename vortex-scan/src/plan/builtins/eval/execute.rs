// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Applying an expression to each batch.

use vortex_array::VortexSessionExecute;
use vortex_error::VortexResult;
use vortex_expr::ArrayExprExt;
use vortex_expr::BoundExpression;
use vortex_functions::variant_get::decode_for_repeated_extraction;

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
    fn compute(&mut self, input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let Input::Batch(batch) = input else {
            return Ok(Step::End);
        };
        let batch = decode_for_repeated_extraction(
            batch,
            &self.expression,
            &mut cx.session().create_execution_ctx(),
        )?;
        Ok(Step::Final(batch.apply_bound(&self.expression)?))
    }
}

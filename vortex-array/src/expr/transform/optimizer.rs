// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::session::{ExprSession, RewriteRuleRegistry};
use crate::expr::transform::simplify::simplify;
use crate::expr::transform::simplify_typed::simplify_typed;

/// A unified optimizer for expressions that can work with or without type information.
#[derive(Debug, Clone)]
pub struct ExprOptimizer {
    rule_registry: RewriteRuleRegistry,
}

impl ExprOptimizer {
    /// Creates a new optimizer with the rules in `ExprSession`.
    pub fn new(session: &ExprSession) -> Self {
        Self {
            rule_registry: session.rewrite_rules().clone(),
        }
    }

    /// Optimize the given expression without a dtype.
    pub fn optimize(&self, expr: Expression) -> VortexResult<Expression> {
        simplify(expr, &self.rule_registry)
    }

    /// Optimize the expression, with a known dtype. This will also apply rules in `optimize`.
    pub fn optimize_typed(&self, expr: Expression, dtype: &DType) -> VortexResult<Expression> {
        simplify_typed(expr, dtype, &self.rule_registry)
    }
}

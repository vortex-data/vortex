// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::transform::ExprOptimizer;

/// Rule context for array rewrite rules
///
/// Provides access to the expression optimizer for optimizing expressions
/// embedded in arrays. Note that dtype is not included since arrays already
/// have a dtype that can be accessed directly.
#[derive(Debug, Clone)]
pub struct ArrayRuleContext {
    expr_optimizer: ExprOptimizer,
}

impl ArrayRuleContext {
    pub fn new(expr_optimizer: ExprOptimizer) -> Self {
        Self { expr_optimizer }
    }

    pub fn expr_optimizer(&self) -> &ExprOptimizer {
        &self.expr_optimizer
    }
}

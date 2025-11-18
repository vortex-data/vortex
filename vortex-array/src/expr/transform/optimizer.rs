// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::session::ExprSession;
use crate::expr::transform::{simplify, simplify_typed};

/// A unified optimizer for expressions that can work with or without type information.
pub struct ExprOptimizer<'a> {
    session: &'a ExprSession,
}

impl<'a> ExprOptimizer<'a> {
    /// Create a new untyped optimizer.
    ///
    /// This optimizer will use untyped simplification rules only.
    pub fn new(session: &'a ExprSession) -> Self {
        Self { session }
    }

    /// Optimize the given expression.
    ///
    /// If this optimizer was created with a dtype, this will perform typed optimization.
    /// Otherwise, it will perform untyped optimization.
    pub fn optimize(&self, expr: Expression) -> VortexResult<Expression> {
        simplify(expr, self.session)
    }

    /// Apply optimize rules to the expression, with a known dtype. This will also apply rules
    /// in `optimize`.
    pub fn optimize_typed(&self, expr: Expression, dtype: &DType) -> VortexResult<Expression> {
        simplify_typed(expr, dtype, self.session)
    }
}

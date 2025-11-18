// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::session::ExprSession;
use crate::expr::transform::{simplify, simplify_typed};

/// A unified optimizer for expressions that can work with or without type information.
///
/// This provides a convenient API for optimizing expressions, automatically choosing between
/// typed and untyped optimization based on whether a dtype is provided.
///
/// # Examples
///
/// ```rust
/// # use vortex_array::expr::Expression;
/// # use vortex_array::expr::session::ExprSession;
/// # use vortex_array::expr::transform::ExprOptimizer;
/// # use vortex_dtype::DType;
/// # fn example(expr: Expression, dtype: DType) {
/// let session = ExprSession::default();
///
/// // Untyped optimization
/// let optimizer = ExprOptimizer::new(&session);
/// let optimized = optimizer.optimize(expr.clone()).unwrap();
///
/// // Typed optimization
/// let optimizer = ExprOptimizer::with_dtype(&session, dtype);
/// let optimized = optimizer.optimize(expr).unwrap();
/// # }
/// ```
pub struct ExprOptimizer<'a> {
    session: &'a ExprSession,
    dtype: Option<DType>,
}

impl<'a> ExprOptimizer<'a> {
    /// Create a new untyped optimizer.
    ///
    /// This optimizer will use untyped simplification rules only.
    pub fn new(session: &'a ExprSession) -> Self {
        Self {
            session,
            dtype: None,
        }
    }

    /// Create a new typed optimizer with the given dtype.
    ///
    /// This optimizer will use both typed and untyped simplification rules,
    /// with access to dtype information.
    pub fn with_dtype(session: &'a ExprSession, dtype: DType) -> Self {
        Self {
            session,
            dtype: Some(dtype),
        }
    }

    /// Optimize the given expression.
    ///
    /// If this optimizer was created with a dtype, this will perform typed optimization.
    /// Otherwise, it will perform untyped optimization.
    pub fn optimize(&self, expr: Expression) -> VortexResult<Expression> {
        match &self.dtype {
            Some(dtype) => simplify_typed(expr, dtype, self.session),
            None => simplify(expr, self.session),
        }
    }

    /// Get the dtype associated with this optimizer, if any.
    pub fn dtype(&self) -> Option<&DType> {
        self.dtype.as_ref()
    }

    /// Get the session associated with this optimizer.
    pub fn session(&self) -> &ExprSession {
        self.session
    }
}

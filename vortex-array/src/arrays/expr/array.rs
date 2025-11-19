// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_ensure};

use crate::expr::Expression;
use crate::stats::ArrayStats;
use crate::{Array, ArrayRef};

/// A array that represents an expression to be evaluated over a child array.
///
/// `ExprArray` enables deferred evaluation of expressions by wrapping a child array
/// with an expression that operates on it. The expression is not evaluated until the
/// array is canonicalized/executed.
///
/// # Examples
///
/// ```ignore
/// // Create an expression that filters an integer array
/// let data = PrimitiveArray::from_iter([1, 2, 3, 4, 5]);
/// let expr = gt(root(), lit(3)); // $ > 3
/// let expr_array = ExprArray::new_infer_dtype(data.into_array(), expr)?;
///
/// // The expression is evaluated when canonicalized
/// let result = expr_array.to_canonical(); // Returns BoolArray([false, false, false, true, true])
/// ```
///
/// # Type Safety
///
/// The `dtype` field must match `expr.return_dtype(child.dtype())`. This invariant
/// is enforced by the safe constructors ([`try_new`](ExprArray::try_new) and
/// [`new_infer_dtype`](ExprArray::new_infer_dtype)) but can be bypassed
/// with [`unchecked_new`](ExprArray::unchecked_new) for performance-critical code.
#[derive(Clone, Debug)]
pub struct ExprArray {
    /// The underlying array that the expression will operate on.
    pub(super) child: ArrayRef,
    /// The expression to evaluate over the child array.
    pub(super) expr: Expression,
    /// The data type of the result after evaluating the expression.
    pub(super) dtype: DType,
    /// Statistics about the resulting array (may be computed lazily).
    pub(super) stats: ArrayStats,
}

impl ExprArray {
    /// Creates a new ExprArray with the dtype validated to match the expression's return type.
    pub fn try_new(child: ArrayRef, expr: Expression, dtype: DType) -> VortexResult<Self> {
        let expected_dtype = expr.return_dtype(child.dtype())?;
        vortex_ensure!(
            dtype == expected_dtype,
            "ExprArray dtype mismatch: expected {}, got {}",
            expected_dtype,
            dtype
        );
        Ok(unsafe { Self::unchecked_new(child, expr, dtype) })
    }

    /// Create a new ExprArray without validating that the dtype matches the expression's return type.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `dtype` matches `expr.return_dtype(child.dtype())`.
    /// Violating this invariant may lead to incorrect results or panics when the array is used.
    pub unsafe fn unchecked_new(child: ArrayRef, expr: Expression, dtype: DType) -> Self {
        Self {
            child,
            expr,
            dtype,
            // TODO(joe): Propagate or compute statistics from the child array and expression.
            stats: ArrayStats::default(),
        }
    }

    /// Creates a new ExprArray with the dtype inferred from the expression and child.
    pub fn new_infer_dtype(child: ArrayRef, expr: Expression) -> VortexResult<Self> {
        let dtype = expr.return_dtype(child.dtype())?;
        Ok(unsafe { Self::unchecked_new(child, expr, dtype) })
    }

    pub fn child(&self) -> &ArrayRef {
        &self.child
    }

    pub fn expr(&self) -> &Expression {
        &self.expr
    }
}

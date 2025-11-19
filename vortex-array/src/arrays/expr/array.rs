// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::stats::ArrayStats;
use crate::{Array, ArrayRef};

#[derive(Clone, Debug)]
pub struct ExprArray {
    pub(super) child: ArrayRef,
    pub(super) expr: Expression,
    pub(super) dtype: DType,
    pub(super) stats: ArrayStats,
}

impl ExprArray {
    pub fn try_new(child: ArrayRef, expr: Expression, dtype: DType) -> VortexResult<Self> {
        assert_eq!(dtype, expr.return_dtype(child.dtype())?);
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
            stats: ArrayStats::default(),
        }
    }

    pub fn new_with_root_dtype(child: ArrayRef, expr: Expression) -> VortexResult<Self> {
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

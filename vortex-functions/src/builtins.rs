// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder methods for this crate's scalar functions, complementing
//! [`ExprBuiltins`](vortex_expr::ExprBuiltins) and
//! [`ArrayBuiltins`](vortex_array::builtins::ArrayBuiltins).

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_error::VortexResult;
use vortex_expr::Expression;
use vortex_expr::ScalarFnExprExt;

use crate::between::Between;
use crate::between::BetweenOptions;
use crate::list_contains::ListContains;
use crate::list_contains::ListContainsOptions;
use crate::zip::Zip;

/// Builder methods on [`Expression`] for this crate's scalar functions.
pub trait ExprFunctions: Sized {
    /// Conditional selection: `result[i] = if mask[i] then if_true[i] else if_false[i]`.
    fn zip(&self, if_true: Expression, if_false: Expression) -> VortexResult<Expression>;

    /// Check if a list contains a value.
    fn list_contains(&self, value: Expression) -> VortexResult<Expression>;
}

impl ExprFunctions for Expression {
    fn zip(&self, if_true: Expression, if_false: Expression) -> VortexResult<Expression> {
        Zip.try_new_expr(EmptyOptions, [if_true, if_false, self.clone()])
    }

    fn list_contains(&self, value: Expression) -> VortexResult<Expression> {
        ListContains.try_new_expr(ListContainsOptions::default(), [self.clone(), value])
    }
}

/// Builder methods on [`ArrayRef`] for this crate's scalar functions.
pub trait ArrayFunctions: Sized {
    /// Conditional selection: `result[i] = if mask[i] then if_true[i] else if_false[i]`.
    fn zip(&self, if_true: ArrayRef, if_false: ArrayRef) -> VortexResult<ArrayRef>;

    /// Check if a list contains a value.
    fn list_contains(&self, value: ArrayRef) -> VortexResult<ArrayRef>;

    /// Compare a values between lower </<= value </<= upper
    fn between(
        self,
        lower: ArrayRef,
        upper: ArrayRef,
        options: BetweenOptions,
    ) -> VortexResult<ArrayRef>;
}

impl ArrayFunctions for ArrayRef {
    fn zip(&self, if_true: ArrayRef, if_false: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(Zip::try_new(if_true, if_false, self.clone())?.into_array())
    }

    fn list_contains(&self, value: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(ListContains::try_new(self.clone(), value)?.into_array())
    }

    fn between(
        self,
        lower: ArrayRef,
        upper: ArrayRef,
        options: BetweenOptions,
    ) -> VortexResult<ArrayRef> {
        Ok(Between::try_new(self, lower, upper, options)?.into_array())
    }
}

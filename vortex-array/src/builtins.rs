// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A collection of built-in common scalar functions.
//!
//! It is expected that each Vortex integration may provide its own set of scalar functions with
//! semantics that exactly match the underlying system (e.g. SQL engine, DataFrame library, etc).
//!
//! This set of functions should cover the basics, and in general leans towards the semantics of
//! the equivalent Arrow compute function.

use vortex_dtype::DType;
use vortex_dtype::FieldName;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArrayExt;
use crate::expr::Between;
use crate::expr::BetweenOptions;
use crate::expr::Cast;
use crate::expr::EmptyOptions;
use crate::expr::Expression;
use crate::expr::FillNull;
use crate::expr::GetItem;
use crate::expr::IsNull;
use crate::expr::ListContains;
use crate::expr::Mask;
use crate::expr::Not;
use crate::expr::VTableExt;
use crate::expr::Zip;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;

/// A collection of built-in scalar functions that can be applied to expressions or arrays.
pub trait ExprBuiltins: Sized {
    /// Cast to the given data type.
    fn cast(&self, dtype: DType) -> VortexResult<Expression>;

    /// Replace null values with the given fill value.
    fn fill_null(&self, fill_value: Expression) -> VortexResult<Expression>;

    /// Get item by field name (for struct types).
    fn get_item(&self, field_name: impl Into<FieldName>) -> VortexResult<Expression>;

    /// Is null check.
    fn is_null(&self) -> VortexResult<Expression>;

    /// Mask the expression using the given boolean mask.
    /// The resulting expression's validity is the intersection of the original expression's
    /// validity.
    fn mask(&self, mask: Expression) -> VortexResult<Expression>;

    /// Boolean negation.
    fn not(&self) -> VortexResult<Expression>;

    /// Check if a list contains a value.
    fn list_contains(&self, value: Expression) -> VortexResult<Expression>;

    /// Conditional selection: `result[i] = if mask[i] then self[i] else if_false[i]`.
    fn zip(&self, if_false: Expression, mask: Expression) -> VortexResult<Expression>;
}

impl ExprBuiltins for Expression {
    fn cast(&self, dtype: DType) -> VortexResult<Expression> {
        Cast.try_new_expr(dtype, [self.clone()])
    }

    fn fill_null(&self, fill_value: Expression) -> VortexResult<Expression> {
        FillNull.try_new_expr(EmptyOptions, [self.clone(), fill_value])
    }

    fn get_item(&self, field_name: impl Into<FieldName>) -> VortexResult<Expression> {
        GetItem.try_new_expr(field_name.into(), [self.clone()])
    }

    fn is_null(&self) -> VortexResult<Expression> {
        IsNull.try_new_expr(EmptyOptions, [self.clone()])
    }

    fn mask(&self, mask: Expression) -> VortexResult<Expression> {
        Mask.try_new_expr(EmptyOptions, [self.clone(), mask])
    }

    fn not(&self) -> VortexResult<Expression> {
        Not.try_new_expr(EmptyOptions, [self.clone()])
    }

    fn list_contains(&self, value: Expression) -> VortexResult<Expression> {
        ListContains.try_new_expr(EmptyOptions, [self.clone(), value])
    }

    fn zip(&self, if_false: Expression, mask: Expression) -> VortexResult<Expression> {
        Zip.try_new_expr(EmptyOptions, [self.clone(), if_false, mask])
    }
}

pub trait ArrayBuiltins: Sized {
    /// Cast to the given data type.
    fn cast(&self, dtype: DType) -> VortexResult<ArrayRef>;

    /// Replace null values with the given fill value.
    fn fill_null(&self, fill_value: impl Into<Scalar>) -> VortexResult<ArrayRef>;

    /// Get item by field name (for struct types).
    fn get_item(&self, field_name: impl Into<FieldName>) -> VortexResult<ArrayRef>;

    /// Is null check.
    fn is_null(&self) -> VortexResult<ArrayRef>;

    /// Mask the array using the given boolean mask.
    /// The resulting array's validity is the intersection of the original array's validity
    /// and the mask's validity.
    fn mask(self, mask: ArrayRef) -> VortexResult<ArrayRef>;

    /// Boolean negation.
    fn not(&self) -> VortexResult<ArrayRef>;

    /// Conditional selection: `result[i] = if mask[i] then self[i] else if_false[i]`.
    fn zip(&self, if_false: ArrayRef, mask: ArrayRef) -> VortexResult<ArrayRef>;

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

impl ArrayBuiltins for ArrayRef {
    fn cast(&self, dtype: DType) -> VortexResult<ArrayRef> {
        if self.dtype() == &dtype {
            return Ok(self.clone());
        }
        Cast.try_new_array(self.len(), dtype, [self.clone()])?
            .optimize()
    }

    fn fill_null(&self, fill_value: impl Into<Scalar>) -> VortexResult<ArrayRef> {
        FillNull
            .try_new_array(
                self.len(),
                EmptyOptions,
                [
                    self.clone(),
                    ConstantArray::new(fill_value.into(), self.len()).into_array(),
                ],
            )?
            .optimize()
    }

    fn get_item(&self, field_name: impl Into<FieldName>) -> VortexResult<ArrayRef> {
        GetItem
            .try_new_array(self.len(), field_name.into(), [self.clone()])?
            .optimize()
    }

    fn is_null(&self) -> VortexResult<ArrayRef> {
        IsNull
            .try_new_array(self.len(), EmptyOptions, [self.clone()])?
            .optimize()
    }

    fn mask(self, mask: ArrayRef) -> VortexResult<ArrayRef> {
        Mask.try_new_array(self.len(), EmptyOptions, [self, mask])?
            .optimize()
    }

    fn not(&self) -> VortexResult<ArrayRef> {
        Not.try_new_array(self.len(), EmptyOptions, [self.clone()])?
            .optimize()
    }

    fn zip(&self, if_false: ArrayRef, mask: ArrayRef) -> VortexResult<ArrayRef> {
        let scalar_fn =
            Zip.try_new_array(self.len(), EmptyOptions, [self.clone(), if_false, mask])?;
        let mut ctx = ExecutionCtx::new(VortexSession::empty());
        scalar_fn.execute::<ArrayRef>(&mut ctx)
    }

    fn list_contains(&self, value: ArrayRef) -> VortexResult<ArrayRef> {
        ListContains
            .try_new_array(self.len(), EmptyOptions, [self.clone(), value])?
            .optimize()
    }

    fn between(
        self,
        lower: ArrayRef,
        upper: ArrayRef,
        options: BetweenOptions,
    ) -> VortexResult<ArrayRef> {
        Between
            .try_new_array(self.len(), options, [self, lower, upper])?
            .optimize()
    }
}

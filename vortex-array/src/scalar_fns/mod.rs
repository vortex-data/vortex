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

use crate::Array;
use crate::ArrayRef;
use crate::arrays::ScalarFnArrayExt;
use crate::expr::Expression;
use crate::expr::ScalarFnExprExt;
use crate::expr::functions::EmptyOptions;

pub mod binary;
pub mod cast;
pub mod get_item;
pub mod is_null;
pub mod mask;
pub mod not;

/// A collection of built-in scalar functions that can be applied to expressions or arrays.
pub trait ExprBuiltins: Sized {
    /// Cast to the given data type.
    fn cast(&self, dtype: DType) -> VortexResult<Expression>;

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
}

impl ExprBuiltins for Expression {
    fn cast(&self, dtype: DType) -> VortexResult<Expression> {
        cast::CastFn.try_new_expr(dtype, [self.clone()])
    }

    fn get_item(&self, field_name: impl Into<FieldName>) -> VortexResult<Expression> {
        get_item::GetItemFn.try_new_expr(field_name.into(), [self.clone()])
    }

    fn is_null(&self) -> VortexResult<Expression> {
        is_null::IsNullFn.try_new_expr(EmptyOptions, [self.clone()])
    }

    fn mask(&self, mask: Expression) -> VortexResult<Expression> {
        mask::MaskFn.try_new_expr(EmptyOptions, [self.clone(), mask])
    }

    fn not(&self) -> VortexResult<Expression> {
        not::NotFn.try_new_expr(EmptyOptions, [self.clone()])
    }
}

pub trait ArrayBuiltins: Sized {
    /// Cast to the given data type.
    fn cast(&self, dtype: DType) -> VortexResult<ArrayRef>;

    /// Get item by field name (for struct types).
    fn get_item(&self, field_name: impl Into<FieldName>) -> VortexResult<ArrayRef>;

    /// Is null check.
    fn is_null(&self) -> VortexResult<ArrayRef>;

    /// Mask the array using the given boolean mask.
    /// The resulting array's validity is the intersection of the original array's validity
    /// and the mask's validity.
    fn mask(&self, mask: &ArrayRef) -> VortexResult<ArrayRef>;

    /// Boolean negation.
    fn not(&self) -> VortexResult<ArrayRef>;
}

impl ArrayBuiltins for ArrayRef {
    fn cast(&self, dtype: DType) -> VortexResult<ArrayRef> {
        cast::CastFn.try_new_array(self.len(), dtype, [self.clone()])
    }

    fn get_item(&self, field_name: impl Into<FieldName>) -> VortexResult<ArrayRef> {
        get_item::GetItemFn.try_new_array(self.len(), field_name.into(), [self.clone()])
    }

    fn is_null(&self) -> VortexResult<ArrayRef> {
        is_null::IsNullFn.try_new_array(self.len(), EmptyOptions, [self.clone()])
    }

    fn mask(&self, mask: &ArrayRef) -> VortexResult<ArrayRef> {
        mask::MaskFn.try_new_array(self.len(), EmptyOptions, [self.clone(), mask.clone()])
    }

    fn not(&self) -> VortexResult<ArrayRef> {
        not::NotFn.try_new_array(self.len(), EmptyOptions, [self.clone()])
    }
}

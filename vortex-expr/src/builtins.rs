// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expression constructors for the scalar functions vortex-array is built on, as methods.

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::builtin::binary::Binary;
use vortex_array::scalar_fn::builtin::cast::Cast;
use vortex_array::scalar_fn::builtin::fill_null::FillNull;
use vortex_array::scalar_fn::builtin::get_item::GetItem;
use vortex_array::scalar_fn::builtin::is_not_null::IsNotNull;
use vortex_array::scalar_fn::builtin::is_null::IsNull;
use vortex_array::scalar_fn::builtin::mask::Mask;
use vortex_array::scalar_fn::builtin::not::Not;
use vortex_array::scalar_fn::builtin::operators::Operator;
use vortex_error::VortexResult;

use crate::Expression;
use crate::ScalarFnExprExt;

/// Builds expressions over the core scalar functions.
pub trait ExprBuiltins: Sized {
    /// Cast to the given data type.
    fn cast(&self, dtype: DType) -> VortexResult<Expression>;

    /// Replace null values with the given fill value.
    fn fill_null(&self, fill_value: Expression) -> VortexResult<Expression>;

    /// Get item by field name (for struct types).
    fn get_item(&self, field_name: impl Into<FieldName>) -> VortexResult<Expression>;

    /// Is null check.
    fn is_null(&self) -> VortexResult<Expression>;

    /// Is not null check.
    fn is_not_null(&self) -> VortexResult<Expression>;

    /// Mask the expression using the given boolean mask.
    /// The resulting expression's validity is the intersection of the original expression's
    /// validity.
    fn mask(&self, mask: Expression) -> VortexResult<Expression>;

    /// Boolean negation.
    fn not(&self) -> VortexResult<Expression>;

    // TODO(joe): add an `interleave` expression builtin mirroring `ArrayBuiltins::interleave`.

    /// Apply a binary operator to this expression and another.
    fn binary(&self, rhs: Expression, op: Operator) -> VortexResult<Expression>;
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

    fn is_not_null(&self) -> VortexResult<Expression> {
        IsNotNull.try_new_expr(EmptyOptions, [self.clone()])
    }

    fn mask(&self, mask: Expression) -> VortexResult<Expression> {
        Mask.try_new_expr(EmptyOptions, [self.clone(), mask])
    }

    fn not(&self) -> VortexResult<Expression> {
        Not.try_new_expr(EmptyOptions, [self.clone()])
    }

    fn binary(&self, rhs: Expression, op: Operator) -> VortexResult<Expression> {
        Binary.try_new_expr(op, [self.clone(), rhs])
    }
}

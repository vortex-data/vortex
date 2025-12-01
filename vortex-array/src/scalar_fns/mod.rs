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
use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::arrays::ScalarFnArrayExt;
use crate::expr::Expression;
use crate::expr::ScalarFnExprExt;

pub mod cast;
pub mod is_null;
pub mod not;

/// A collection of built-in scalar functions that can be applied to expressions or arrays.
pub trait BuiltinScalarFns: Sized {
    /// Cast to the given data type.
    fn cast(&self, dtype: DType) -> VortexResult<Self>;
}

impl BuiltinScalarFns for Expression {
    fn cast(&self, dtype: DType) -> VortexResult<Expression> {
        cast::CastFn.try_new_expr(dtype, [self.clone()])
    }
}

impl BuiltinScalarFns for ArrayRef {
    fn cast(&self, dtype: DType) -> VortexResult<Self> {
        cast::CastFn.try_new_array(self.len(), dtype, [self.clone()])
    }
}

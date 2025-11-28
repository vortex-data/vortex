// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A temporary module for evaluating expressions against arrays while we migrate to lazy
//! execution framework.
//!
//! This module stores expression evaluation logic in the VortexSession.

use crate::expr::functions::scalar::ScalarFn;
use crate::expr::functions::FunctionId;
use crate::expr::{functions, ExprId, Expression, ExpressionView};
use crate::{expr, ArrayRef};
use std::any::Any;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

/// Evaluate an expression against a Vortex array.
///
/// For now, the evaluation logic preserves the existing semi-eager execution model.
/// In the future, this will be replaced by a fully lazy execution framework, where
/// expressions can be applied to an array in essentially constant time, with optimization rules
/// performing subsequent push-down.
#[derive(Default, Clone)]
pub struct ArrayEvaluator {
    evaluations: HashMap<ExprId, Arc<dyn DynArrayExprEvaluation>>,
    fn_evaluations: HashMap<FunctionId, Arc<dyn DynArrayFnEvaluation>>,
}

impl Debug for ArrayEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayEvaluator")
            .field("evaluations", &self.evaluations.keys())
            .finish()
    }
}

impl ArrayEvaluator {
    /// Register the evaluation logic for the given expression ID.
    pub fn register<V: expr::VTable>(
        &mut self,
        vtable: &'static V,
        evaluator: impl ArrayExprEvaluation<V>,
    ) {
        self.evaluations.insert(
            vtable.id(),
            Arc::new(ArrayExprEvaluationAdapter {
                inner: evaluator,
                _marker: PhantomData,
            }),
        );
    }

    /// Register the evaluation logic for the given scalar function.
    pub fn register_fn<V: functions::VTable>(
        &mut self,
        vtable: &'static V,
        evaluator: impl ArrayFnEvaluation<V>,
    ) {
        self.fn_evaluations.insert(
            vtable.id(),
            Arc::new(ArrayFnEvaluationAdapter {
                inner: evaluator,
                _marker: PhantomData,
            }),
        );
    }

    /// Evaluate an expression against an array.
    pub fn evaluate(&self, expression: &Expression, array: &ArrayRef) -> VortexResult<ArrayRef> {
        let evaluator = self
            .evaluations
            .get(&expression.vtable().id())
            .ok_or_else(|| {
                vortex_error::vortex_err!(
                    "No evaluator registered for expression ID {}",
                    expression.vtable().id()
                )
            })?;
        evaluator.evaluate(expression, array)
    }

    /// Evaluate a scalar function against its input arrays.
    pub fn evaluate_fn(&self, function: &ScalarFn, arrays: &[ArrayRef]) -> VortexResult<ArrayRef> {
        let evaluator = self.fn_evaluations.get(&function.id()).ok_or_else(|| {
            vortex_error::vortex_err!("No evaluator registered for function ID {}", function.id())
        })?;
        evaluator.evaluate(function, arrays)
    }
}

/// A plugin trait for evaluating scalar functions against arrays.
pub trait ArrayFnEvaluation<V: functions::VTable>: 'static + Send + Sync {
    fn evaluate(&self, options: &V::Options, args: &[ArrayRef]) -> VortexResult<ArrayRef>;
}

trait DynArrayFnEvaluation: 'static + Send + Sync {
    fn evaluate(&self, options: &dyn Any, args: &[ArrayRef]) -> VortexResult<ArrayRef>;
}

struct ArrayFnEvaluationAdapter<V: functions::VTable, E: ArrayFnEvaluation<V>> {
    inner: E,
    _marker: PhantomData<V>,
}

impl<V: functions::VTable, E: ArrayFnEvaluation<V>> DynArrayFnEvaluation
    for ArrayFnEvaluationAdapter<V, E>
{
    fn evaluate(&self, options: &dyn Any, args: &[ArrayRef]) -> VortexResult<ArrayRef> {
        let opts = options.downcast_ref::<V::Options>().ok_or_else(|| {
            vortex_error::vortex_err!("Invalid options type for function evaluation")
        })?;
        self.inner.evaluate(opts, args)
    }
}

/// A plugin trait for evaluating expressions against arrays.
pub trait ArrayExprEvaluation<V: expr::VTable>: 'static + Send + Sync {
    /// Evaluate an expression against an array.
    fn evaluate(&self, expression: &ExpressionView<V>, array: &ArrayRef) -> VortexResult<ArrayRef>;
}

/// A plugin trait for evaluating expressions against arrays.
trait DynArrayExprEvaluation: 'static + Send + Sync {
    /// Evaluate an expression against an array.
    fn evaluate(&self, expression: &Expression, array: &ArrayRef) -> VortexResult<ArrayRef>;
}

struct ArrayExprEvaluationAdapter<V: expr::VTable, E: ArrayExprEvaluation<V>> {
    inner: E,
    _marker: PhantomData<V>,
}

impl<V: expr::VTable, E: ArrayExprEvaluation<V>> DynArrayExprEvaluation
    for ArrayExprEvaluationAdapter<V, E>
{
    fn evaluate(&self, expression: &Expression, array: &ArrayRef) -> VortexResult<ArrayRef> {
        let expr_view = ExpressionView::<V>::new(expression);
        self.inner.evaluate(&expr_view, array)
    }
}

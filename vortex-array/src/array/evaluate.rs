// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A temporary module for evaluating expressions against arrays while we migrate to lazy
//! execution framework.
//!
//! This module stores expression evaluation logic in the VortexSession.

use crate::expr::{ExprId, Expression};
use crate::ArrayRef;
use std::fmt::Debug;
use std::sync::Arc;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

/// Evaluate an expression against a Vortex array.
///
/// For now, the evaluation logic preserves the existing semi-eager execution model.
/// In the future, this will be replaced by a fully lazy execution framework, where
/// expressions can be applied to an array in essentially constant time, with optimization rules
/// performing subsequent push-down.
#[derive(Default, Debug)]
pub struct ArrayEvaluator {
    evaluations: HashMap<ExprId, Arc<dyn ArrayExprEvaluation>>,
}

/// A plugin trait for evaluating expressions against arrays.
pub trait ArrayExprEvaluation: 'static + Send + Sync + Debug {
    /// Evaluate an expression against an array.
    fn evaluate(&self, expression: &Expression, array: &ArrayRef) -> VortexResult<ArrayRef>;
}

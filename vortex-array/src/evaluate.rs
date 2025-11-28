// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A temporary module for evaluating expressions against arrays while we migrate to lazy
//! execution framework.
//!
//! This module stores expression evaluation logic in the VortexSession.

use crate::expr::Expression;
use crate::ArrayRef;
use vortex_error::VortexResult;

pub trait ExprArrayEvaluation {
    fn evaluate_array(&self, array: ArrayRef) -> VortexResult<ArrayRef>;
}

impl ExprArrayEvaluation for Expression {
    fn evaluate_array(&self, array: ArrayRef) -> VortexResult<ArrayRef> {
        todo!()
    }
}

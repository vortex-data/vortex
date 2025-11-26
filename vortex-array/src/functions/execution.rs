// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the API for vectorized execution within Vortex.
// TODO(ngates): these definitions should be lifted out of the functions module.

use crate::ArrayRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_vector::Vector;

/// Context provided when executing an object using vectorized execution.
pub trait ExecutionCtx {
    /// The number of rows to be processed.
    fn row_count(&self) -> usize;
    /// The expected return dtype of the execution.
    fn return_dtype(&self) -> DType;
    /// The data type of nth input.
    fn input_dtype(&self, input_idx: usize) -> DType;
    /// The vector of the nth input.
    fn input_vector(&self, input_idx: usize) -> VortexResult<Vector>;
}

/// Legacy context for evaluating compute functions over arrays.
pub trait EvaluationCtx {
    /// The number of rows to be processed.
    fn row_count(&self) -> usize;
    /// The expected return dtype of the evaluation.
    fn return_dtype(&self) -> DType;
    /// The array of the nth input.
    fn input_array(&self, input_idx: usize) -> VortexResult<ArrayRef>;
}

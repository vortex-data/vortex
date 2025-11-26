// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the API for vectorized execution within Vortex.
// TODO(ngates): these definitions should be lifted out of the functions module.

use vortex_dtype::DType;
use vortex_vector::Vector;
use vortex_vector::VectorOps;

/// Context provided when executing an object using vectorized execution.
pub struct ExecutionCtx {
    /// The number of rows to be processed.
    row_count: usize,
    /// The expected return dtype of the execution.
    return_dtype: DType,
    /// The input data types.
    input_types: Vec<DType>,
    /// The input vectors.
    input_vectors: Vec<Vector>,
}

impl ExecutionCtx {
    pub fn new(
        row_count: usize,
        return_dtype: DType,
        input_types: Vec<DType>,
        input_vectors: Vec<Vector>,
    ) -> Self {
        assert!(
            input_vectors.iter().all(|v| v.len() == row_count),
            "All input vectors must have the same length as row_count"
        );
        Self {
            row_count,
            return_dtype,
            input_types,
            input_vectors,
        }
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn return_dtype(&self) -> &DType {
        &self.return_dtype
    }

    pub fn input_type(&self, idx: usize) -> &DType {
        &self.input_types[idx]
    }

    pub fn input_vectors(&self, idx: usize) -> &Vector {
        &self.input_vectors[idx]
    }
}

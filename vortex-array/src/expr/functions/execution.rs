// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the API for vectorized execution within Vortex.
// TODO(ngates): these definitions should be lifted out of the functions module.

use vortex_dtype::DType;
use vortex_vector::Datum;
use vortex_vector::VectorOps;

/// Context provided when executing an object using vectorized execution.
pub struct ExecutionCtx {
    /// The number of rows to be processed.
    row_count: usize,
    /// The expected return dtype of the execution.
    return_dtype: DType,
    /// The input data types.
    input_dtypes: Vec<DType>,
    /// The input datums.
    input_datums: Vec<Datum>,
}

impl ExecutionCtx {
    pub fn new(
        row_count: usize,
        return_dtype: DType,
        input_types: Vec<DType>,
        input_datums: Vec<impl Into<Datum>>,
    ) -> Self {
        let input_datums: Vec<Datum> = input_datums.into_iter().map(|d| d.into()).collect();
        assert!(
            input_datums
                .iter()
                .all(|d| d.as_vector().is_none_or(|v| v.len() == row_count)),
            "All input vectors must have the same length as row_count"
        );
        Self {
            row_count,
            return_dtype,
            input_dtypes: input_types,
            input_datums,
        }
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn return_dtype(&self) -> &DType {
        &self.return_dtype
    }

    pub fn input_type(&self, idx: usize) -> &DType {
        &self.input_dtypes[idx]
    }

    pub fn input_datums(&self, idx: usize) -> &Datum {
        &self.input_datums[idx]
    }
}

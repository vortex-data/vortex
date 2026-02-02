// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA-accelerated execution for the DataFusion benchmark.
//!
//! This module provides CUDA-accelerated projection execution for TPC-H benchmarks.
//! It duplicates the opener logic from vortex-datafusion but uses CUDA execution
//! instead of CPU execution.

mod format;
mod opener;
mod source;

pub use format::CudaVortexFormat;
pub use source::CudaVortexSource;

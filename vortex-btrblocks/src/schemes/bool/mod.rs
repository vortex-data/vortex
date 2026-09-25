// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bool compression schemes.

mod sparse;

pub use sparse::SparseScheme;
pub use vortex_compressor::stats::BoolStats;

#[cfg(test)]
mod tests;

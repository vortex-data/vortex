// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Field-access analyses over bound expressions, used to prune the columns a scan reads.
//!
//! They look through struct projections (`get_item` and `select`) to the fields of the scan's
//! root scope.

mod immediate_access;
mod referenced_field_paths;

pub use immediate_access::*;
pub use referenced_field_paths::referenced_field_paths;

#[cfg(test)]
mod tests;

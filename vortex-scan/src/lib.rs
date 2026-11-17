// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! The Vortex Scan API implements an abstract table scan interface that can be used to
//! read data from various data sources.
//!
//! It supports arbitrary projection expressions, filter expressions, and limit pushdown as well
//! as mechanisms for parallel and distributed execution via partitions.
//!
//! The API is currently under development and may change in future releases, however we hope to
//! stabilize into stable C ABI for use within foreign language bindings.
//!
//! If you are looking to scan Vortex files or layouts, the Vortex implementation of the Scan API
//! is lowered by `vortex-layout` into this crate's plans and executor.
//!
//! ## Open Issues
//!
//! * We probably want to make the DataSource serializable as well, so that we can share
//!   source-level state with workers, separately from partition serialization.
//! * We should add a way for the client to negotiate capabilities with the data source, for
//!   example which encodings it knows about.

pub mod arrays;
pub mod execution;
pub mod plan;
pub mod planning;
pub mod prepare;
pub mod request;
pub mod source;

pub use request::ScanRequest;
pub mod row_mask;
pub mod selection;
pub mod strict_sorted_buffer;

pub use source::DataSource;
pub use source::DataSourceOpener;
pub use source::DataSourceRef;
pub use source::DataSourceRemote;
pub use source::DataSourceScan;
pub use source::DataSourceScanRef;
pub use source::Partition;
pub use source::PartitionRef;
pub use source::PartitionStream;

/// Expressions referring to the global scan row index.
pub mod row_idx {
    pub use crate::plan::builtins::row_idx::expr::*;
}

#[cfg(test)]
mod test;

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An experimental implementation of the Vortex Scan API for DataFusion.
//!
//! This integration directly implements `TableProvider` + `ExecutionPlan`, bypassing DataFusion's
//! `FileFormat` abstraction. Instead, we prefer to build out Vortex's MultiFileDataSource in order
//! to provide the same level of functionality across all query engines.

mod exec;
mod table;

pub use exec::VortexExec;
pub use table::VortexTable;

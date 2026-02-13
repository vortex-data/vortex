// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An experimental implementation of the Vortex Scan API for DataFusion.
//!
//! This integration directly implements `TableProvider` + `ExecutionPlan`, bypassing DataFusion's
//! `FileFormat` abstraction.

mod cache;
mod exec;
mod table;

pub use cache::DataFusionFooterCache;
pub use exec::VortexExec;
pub use table::VortexTable;

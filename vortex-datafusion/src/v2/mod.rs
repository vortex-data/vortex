// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An experimental implementation of the Vortex Scan API for DataFusion.
//!
//! This integration directly implements `TableProvider` + `ExecutionPlan`, bypassing DataFusion's
//! `FileFormat` abstraction.

mod source;
mod table;

pub use source::VortexDataSource;
pub use table::VortexTable;

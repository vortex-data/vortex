// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Direct DataFusion integration for an existing Vortex
//! [`DataSourceRef`].
//!
//! Use this module when some other part of the system has already selected the
//! Vortex source to query and DataFusion only needs an adapter around it.
//!
//! Typical flow:
//!
//! 1. Build or obtain a Vortex [`DataSourceRef`].
//! 2. Wrap it in [`VortexTable`] to register it with a [`SessionContext`], or
//!    build a [`VortexDataSource`] directly when constructing a
//!    [`DataSourceExec`].
//! 3. Let DataFusion apply projection, filter, and limit pushdown through the
//!    resulting adapter.
//!
//! The two main types are:
//!
//! - [`VortexTable`], the higher-level
//!   [`TableProvider`] for `SessionContext::register_table`.
//! - [`VortexDataSource`], the lower-level
//!   [`DataSource`] used when constructing physical plans directly.
//!
//! Compared with [`crate::VortexFormatFactory`], this module starts from an
//! already-constructed Vortex source instead of asking DataFusion to discover
//! `.vortex` files.
//!
//! [`DataSourceRef`]: vortex::scan::DataSourceRef
//! [`SessionContext`]: https://docs.rs/datafusion/latest/datafusion/prelude/struct.SessionContext.html
//! [`DataSourceExec`]: datafusion_datasource::source::DataSourceExec
//! [`TableProvider`]: datafusion_catalog::TableProvider
//! [`DataSource`]: datafusion_datasource::source::DataSource

mod source;
mod table;

pub use source::VortexDataSource;
pub use table::VortexTable;

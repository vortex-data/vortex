// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! File-backed DataFusion integration for `.vortex` files.
//!
//! Use this module when Vortex data lives in a filesystem or object store and
//! you want DataFusion to discover files, infer schema, and build
//! [`DataSourceExec`] plans for you.
//!
//! The main entry points are:
//!
//! - [`VortexFormatFactory`] to register the `vortex` file format with
//!   DataFusion.
//! - [`VortexFormat`] when constructing
//!   [`ListingOptions`] directly.
//! - [`VortexSource`] for lower-level `FileScanConfig` construction.
//! - [`VortexAccessPlan`] when external indexing or custom file selection needs
//!   to restrict what a scan reads.
//! - [`metrics::VortexMetricsFinder`] to collect Vortex-specific scan metrics
//!   from a planned query.
//!
//! [`DataSourceExec`]: datafusion_datasource::source::DataSourceExec
//! [`ListingOptions`]: https://docs.rs/datafusion/latest/datafusion/datasource/listing/struct.ListingOptions.html

mod access_plan;
mod cache;
mod format;
pub mod metrics;
mod opener;
pub mod reader;
mod sink;
mod source;
mod stream;

pub use access_plan::VortexAccessPlan;
pub use format::VortexFormat;
pub use format::VortexFormatFactory;
pub use format::VortexTableOptions;
pub use sink::VortexSink;
pub use source::VortexSource;

#[cfg(test)]
mod tests;

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Multi-file data source for scanning across multiple Vortex files.
//!
//! This module provides [`MultiFileDataSource`], a reusable, engine-agnostic data source that
//! discovers and opens multiple Vortex files, presenting them as a single [`DataSource`]
//! for scanning. It is analogous to DataFusion's `ListingTable`.
//!
//! Use [`MultiFileDataSourceBuilder`] to construct a `MultiFileDataSource` from an object store
//! and a set of file paths or a glob pattern.
//!
//! [`DataSource`]: vortex_scan::api::DataSource

mod builder;
mod glob;
mod source;

pub use builder::FileDiscovery;
pub use builder::MultiFileDataSourceBuilder;
pub use builder::SchemaResolution;
pub use source::MultiFileDataSource;

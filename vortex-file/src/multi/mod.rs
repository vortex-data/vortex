// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Multi-file data source for scanning across multiple Vortex files.
//!
//! [`MultiFileDataSource`] discovers and opens multiple Vortex files from a glob pattern,
//! presenting them as a single [`DataSource`](vortex_scan::api::DataSource) for scanning.
//! Footer caching is handled automatically via the session's [`MultiFileSession`].
//!
//! ```ignore
//! let ds = MultiFileDataSource::new(session)
//!     .with_glob_url("/data/*.vortex")
//!     .build()
//!     .await?;
//! ```

mod builder;
pub mod session;

pub use builder::MultiFileDataSource;

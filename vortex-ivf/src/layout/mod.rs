// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! IVF layout: on-disk format for IVF-indexed vector columns.
//!
//! An [`IvfLayout`] wraps a chunked data layout (where each chunk is one cluster) with an
//! auxiliary centroid child. At read time, the [`IvfReader`] inspects the filter expression
//! and, if it contains a cosine similarity against a constant query vector, uses the centroids
//! to prune out chunks whose cluster centroid is far from the query.
//!
//! # Layout tree
//!
//! ```text
//! IvfLayout
//!  ├── data       (Transparent)  ChunkedLayout { chunk 0 = cluster 0, chunk 1 = cluster 1, ... }
//!  └── centroids  (Auxiliary)    FlatLayout { Vector<dim, f32>, one row per cluster }
//! ```
//!
//! # Pruning
//!
//! When the scan presents an expression like `CosineSimilarity(root, const_query) > threshold`,
//! the reader:
//!
//! 1. Fetches the centroids child (small, one-off read).
//! 2. Computes cosine similarity between the query and each cluster centroid.
//! 3. Selects the top `nprobes` clusters.
//! 4. Returns a pruning mask that is `false` for every row in a non-probed cluster.
//!
//! This leverages the existing chunked data infrastructure and TurboQuant-compatible expression
//! fast paths. The IVF layer does not alter how individual rows are scanned — it only decides
//! which chunks to open.

mod metadata;
mod query;
mod reader;
mod vtable;
pub mod writer;

#[cfg(test)]
mod tests;

pub use metadata::IvfLayoutMetadata;
pub use vtable::Ivf;
pub use vtable::IvfLayout;
pub use vtable::IvfLayoutEncoding;

/// The layout encoding ID for [`IvfLayout`].
pub const IVF_LAYOUT_ID: &str = "vortex.ivf";

/// Default number of clusters to probe at query time.
pub const DEFAULT_NPROBES: u32 = 8;

use vortex_layout::LayoutEncodingRef;
use vortex_layout::session::LayoutSessionExt;
use vortex_session::VortexSession;

/// Register [`IvfLayoutEncoding`] with the given session so IVF-encoded Vortex files can be read.
///
/// Applications that want to write or read files containing [`IvfLayout`] must call this once
/// against the session they'll use.
pub fn register_ivf_layout(session: &VortexSession) {
    session
        .layouts()
        .register(LayoutEncodingRef::new_ref(IvfLayoutEncoding.as_ref()));
}

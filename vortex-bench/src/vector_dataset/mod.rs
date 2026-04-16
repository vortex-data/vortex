// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Public catalog of VectorDBBench corpora used by the vector-search benchmark.
//!
//! The catalog is intentionally separate from the [`crate::datasets::Dataset`] trait used by the
//! row-table benchmarks: the train split of a vector dataset is sometimes partitioned across many
//! parquet files, sometimes single-file, sometimes shuffled, sometimes not, and its `emb` column
//! has to be rewrapped into an `Extension<Vector>` before it's useful to a cosine-similarity scan.
//! None of that fits the row-table `Dataset` contract.
//!
//! The four sub-modules split the catalog into roughly orthogonal concerns:
//!
//! - `catalog`: the static [`VectorDataset`] enum + per-dataset metadata.
//! - `layout`: [`TrainLayout`] / [`LayoutSpec`] (the four hosted train shapes) and
//!   [`VectorMetric`].
//! - `download`: URL builders and the idempotent download driver.
//! - `paths`: local filesystem layout (`<data_dir>/vector-search/...`).
//!
//! Higher-level callers (the bench crate's ingest + scan pipeline) compose these:
//! [`download::download`] returns a [`download::DatasetPaths`] handle that the ingest pass turns
//! into per-flavor `.vortex` files, after which the scan driver re-opens those files per iteration.

mod catalog;
mod download;
mod layout;
mod paths;

pub use catalog::ALL_VECTOR_DATASETS;
pub use catalog::VectorDataset;
pub use download::DatasetPaths;
pub use download::download;
pub use download::neighbors_url;
pub use download::test_url;
pub use download::train_urls;
pub use layout::LayoutSpec;
pub use layout::TrainLayout;
pub use layout::VectorMetric;

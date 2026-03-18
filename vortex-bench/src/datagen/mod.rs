// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! # Benchmark Dataset Repository Manager
//!
//! ## Motivation
//!
//! Benchmark datasets are large, slow to generate, and shared across
//! developers and CI. Without a central registry, every developer
//! regenerates TPC-H SF100 locally, CI jobs download from ad-hoc URLs,
//! and nobody knows which version of a dataset produced a given benchmark
//! result. This module solves that with:
//!
//! - **Content-addressed storage**: every file is SHA-256 hashed.
//!   Uploads and downloads skip files whose hash already matches,
//!   so re-pushing 100 GB with one changed file only transfers the diff.
//! - **A single catalog**: `catalog.json` at the repo root lists every
//!   dataset with its manifest hash. `pull` fetches just this metadata
//!   (cheap), `checkout` fetches the actual data (expensive).
//! - **Provenance tracking**: `dataset.yaml` records who created the
//!   data, how (generator command, external URL, derived from another
//!   dataset), and arbitrary tags.
//! - **Head sample**: the manifest embeds the first 8 KiB of the first
//!   file so consumers can peek at data (`bench-data head`) without
//!   downloading anything.
//!
//! ## Quick start
//!
//! ```bash
//! # Author: create and publish a dataset
//! bench-data init tpch-sf100
//! vim tpch-sf100/dataset.yaml          # fill in metadata
//! cp *.parquet tpch-sf100/data/parquet/lineitem/
//! bench-data push tpch-sf100/ --remote s3://my-bucket
//!
//! # Consumer: browse and download
//! bench-data pull --remote s3://my-bucket
//! bench-data list
//! bench-data head tpch-sf100           # peek at data (no download)
//! bench-data checkout tpch-sf100 --remote s3://my-bucket
//!
//! # Maintenance
//! bench-data verify tpch-sf100 --remote s3://my-bucket
//! bench-data gc --remote s3://my-bucket
//! ```
//!
//! ## Commands
//!
//! | Command | Network? | Description |
//! |---------|----------|-------------|
//! | `init` | No | Scaffold dataset dir with template `dataset.yaml` |
//! | `manifest` | No | Hash all files in `data/`, write `manifest.json` |
//! | `validate` | No | Check descriptor + data before pushing |
//! | `push` | Yes | Upload to remote (skips unchanged files) |
//! | `pull` | Yes | Fetch catalog + manifests (no data) |
//! | `checkout` | Yes | Download data files (skips cached) |
//! | `list` | No | List datasets from local catalog |
//! | `describe` | No | Show dataset metadata |
//! | `head` | No | Show first 8 KiB sample from manifest |
//! | `delete` | Yes | Remove dataset from catalog (+ optional purge) |
//! | `gc` | Yes | Remove orphaned dirs + stale upload locks |
//! | `verify` | Yes | Check all remote file hashes match manifest |
//!
//! ## Implementation
//!
//! ### Transfer optimization
//!
//! Both push and checkout use the same core abstraction: [`manifest::FileIndex`],
//! a hash map from relative path to `(sha256, size)`.
//!
//! - **Push**: builds a `FileIndex` from the old remote manifest. For each
//!   file in the new manifest, if the hash matches the old index, it uses
//!   `ObjectStore::copy` (server-side, zero egress) instead of re-uploading.
//!   If ALL files match and the file count is identical, it reuses the old
//!   remote path entirely — no data transfer, no orphaned directory.
//!
//! - **Checkout**: for each file in the manifest, calls
//!   [`manifest::file_matches_hash`] on the local copy. Skips download if
//!   the hash matches.
//!
//! ### Concurrency control
//!
//! Push acquires a lock file (`{name}.uploading`) via `PutMode::Create`
//! (atomic create-if-not-exists). The lock contains a timestamp so `gc`
//! can distinguish stale locks (>1h) from active ones. On any exit path
//! (success or error), the lock is released.
//!
//! ### Storage layout
//!
//! ```text
//! s3://bucket/
//! ├── catalog.json                  # top-level index
//! ├── tpch-sf100.uploading          # lock file (only during push)
//! ├── tpch-sf100-m9d2k4/            # dataset directory ({name}-{rand}/)
//! │   ├── dataset.yaml              # human-authored metadata
//! │   ├── manifest.json             # auto-generated file index + head sample
//! │   └── parquet/lineitem/         # data: {format}/{table}/{files}
//! │       ├── lineitem_000.parquet
//! │       └── lineitem_001.parquet
//! └── clickbench-f7k3j9/
//!     └── ...
//! ```
//!
//! The local mirror (`~/.cache/vortex-bench-data/`) uses the same layout.
//! `pull` populates metadata, `checkout` populates `data/`.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                       bench-data CLI                            │
//! │  (bin/bench-data.rs — argument parsing, user prompts, display) │
//! └───────────────┬──────────────────┬──────────────────────────────┘
//!                 │                  │
//!        ┌────────▼───────┐  ┌───────▼────────┐
//!        │   local.rs     │  │   remote.rs     │
//!        │  init          │  │  push / pull    │
//!        │  manifest      │  │  checkout       │
//!        │  validate      │  │  delete / gc    │
//!        └────────┬───────┘  │  verify         │
//!                 │          └───────┬──────────┘
//!        ┌────────▼─────────────────▼──────────┐
//!        │          manifest.rs                 │
//!        │  Manifest, FileEntry, FileIndex      │
//!        │  hash_file, file_matches_hash        │
//!        │  head sample encode/decode           │
//!        ├──────────────────────────────────────┤
//!        │  catalog.rs      │  dataset.rs       │
//!        │  Catalog         │  DatasetDescriptor│
//!        │  DatasetEntry    │  Source            │
//!        └──────────────────┴───────────────────┘
//! ```

pub mod catalog;
pub mod dataset;
pub mod local;
pub mod manifest;
pub mod remote;

#[cfg(test)]
mod tests;

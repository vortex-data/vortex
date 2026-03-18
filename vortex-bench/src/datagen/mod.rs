// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark dataset repository manager.
//!
//! Manages a catalog of benchmark datasets stored in object storage (S3, GCS,
//! or local filesystem). Supports creating, uploading, listing, downloading,
//! and verifying datasets with content-addressed integrity checking.
//!
//! # Concepts
//!
//! - **Catalog**: top-level index of all datasets, stored as `catalog.json`
//! - **Dataset**: a named collection of data files with a human-authored
//!   descriptor (`dataset.yaml`) and auto-generated manifest (`manifest.json`)
//! - **Manifest**: lists all files with their SHA-256 hashes, organized by
//!   format and table
//!
//! # S3 layout
//!
//! ```text
//! s3://bucket/
//! ├── catalog.json
//! ├── tpch-sf100-m9d2k4/
//! │   ├── dataset.yaml
//! │   ├── manifest.json
//! │   └── parquet/
//! │       └── lineitem/
//! │           ├── lineitem_000.parquet
//! │           └── lineitem_001.parquet
//! └── clickbench-f7k3j9/
//!     ├── dataset.yaml
//!     ├── manifest.json
//!     └── parquet/
//!         └── hits/
//!             └── hits.parquet
//! ```
//!
//! Local mirror uses the same layout. `pull` fetches metadata, `checkout`
//! fetches data files.
//!
//! # Usage graph
//!
//! ```text
//!                    ┌──────────────────────────────────────────────┐
//!                    │              Author workflow                 │
//!                    └──────────────────────────────────────────────┘
//!
//!     init ──► edit dataset.yaml ──► add data files ──► push --remote <url>
//!      │                                                     │
//!      ▼                                                     ▼
//!   my-dataset/                                ┌─── acquire lock (CAS) ───┐
//!   ├── dataset.yaml  (you write)              │   {name}.uploading       │
//!   └── data/                                  ▼                          │
//!       └── {format}/{table}/files       upload files                     │
//!                                              │                          │
//!                                              ▼                          │
//!                                        upload manifest.json             │
//!                                              │                          │
//!                                              ▼                          │
//!                                        update catalog.json              │
//!                                              │                          │
//!                                              ▼                          │
//!                                        release lock ◄──────────────────┘
//!
//!                    ┌──────────────────────────────────────────────┐
//!                    │             Consumer workflow                │
//!                    └──────────────────────────────────────────────┘
//!
//!   pull ──────────────────────► checkout <name> ────────► use data
//!     │                               │
//!     ▼                               ▼
//!   fetches catalog.json         downloads data files
//!   + all manifests              (skips if hash matches)
//!   + all dataset.yaml
//!
//!                    ┌──────────────────────────────────────────────┐
//!                    │              Maintenance                     │
//!                    └──────────────────────────────────────────────┘
//!
//!   list ──► describe <name>     show catalog / dataset details
//!   verify <name>                check all file hashes match manifest
//!   delete <name> [--purge]      remove from catalog (optionally purge files)
//!   gc                           remove orphaned directories not in catalog
//! ```

pub mod catalog;
pub mod dataset;
pub mod local;
pub mod manifest;
pub mod remote;

#[cfg(test)]
mod tests;

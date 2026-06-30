// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SpatialBench data preparation. [`wkb`] generates the canonical WKB base tables (Parquet + Vortex);
//! the [`table`] catalog is the single source of truth for the base tables.

pub mod table;
pub mod wkb;

pub use table::Table;
pub use wkb::generate_tables;

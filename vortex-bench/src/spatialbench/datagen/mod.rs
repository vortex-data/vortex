// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SpatialBench data preparation. [`wkb`] generates the canonical WKB base tables; [`native`]
//! derives the native-Point encodings from them for `points=native`. The [`table`] catalog is the
//! single source of truth for the base tables both stages share.

pub mod native;
pub mod table;
pub mod wkb;

pub use native::write_native_parquet;
pub use native::write_native_vortex;
pub use table::Table;
pub use wkb::generate_tables;

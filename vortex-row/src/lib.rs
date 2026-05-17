// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-oriented byte encoder, analogous to Apache Arrow's `arrow-row` crate.
//!
//! Subsequent commits add the encoder, decoder helpers, and per-encoding fast paths.
//! This commit only establishes the crate skeleton and an `initialize` stub.

pub mod options;

pub use options::RowEncodeOptions;
pub use options::SortField;
use vortex_session::VortexSession;

/// Register the row-encoding scalar functions on the given session.
///
/// Currently a stub: subsequent commits register `RowSize` and `RowEncode` here.
pub fn initialize(_session: &VortexSession) {}

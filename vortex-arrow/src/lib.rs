// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Apache Arrow interoperability for Vortex.
//!
//! This crate owns every conversion between Vortex and Arrow. The authoritative entry point is
//! the [`ArrowSession`]: importing Arrow schemas, fields, and data types into Vortex
//! ([`ArrowSession::from_arrow_schema`], [`ArrowSession::from_arrow_field`],
//! [`ArrowSession::from_arrow_datatype`]), importing Arrow arrays and record batches
//! ([`ArrowSession::from_arrow_array`], [`ArrowSession::from_arrow_record_batch`]),
//! exporting Vortex dtypes to Arrow
//! ([`ArrowSession::to_arrow_schema`], [`ArrowSession::to_arrow_field`],
//! [`ArrowSession::to_arrow_datatype`]), and executing Vortex arrays into Arrow
//! ([`ArrowSession::execute_arrow`] and the [`ArrowArrayExecutor`] convenience trait).
//!
//! The [`ArrowSession`] session variable is created lazily on first use of
//! [`ArrowSessionExt::arrow`], so no explicit registration is required for plain conversions.

use vortex_session::VortexSession;

pub mod convert;
mod datum;
pub mod dtype;
mod executor;
mod iter;
mod null_buffer;
mod options;
mod run_end_import;
mod scalar;
mod session;

pub use convert::IntoVortexArray;
pub use convert::nulls;
pub use datum::*;
pub use executor::*;
pub use iter::*;
pub use null_buffer::to_arrow_null_buffer;
pub use null_buffer::to_null_buffer;
pub use options::*;
pub use scalar::ToArrowDatum;
pub use session::*;

/// Register vortex-arrow's session extensions into `session`.
///
/// This eagerly registers the [`ArrowSession`], which is otherwise created lazily on first use.
pub fn initialize(session: &VortexSession) {
    session.register(ArrowSession::default());
}

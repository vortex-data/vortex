// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Apache Arrow interoperability for Vortex.
//!
//! This crate owns every conversion between Vortex and Arrow. Exporting a Vortex array to Arrow goes
//! through [`ExecuteArrowExt`], which uses the [`ArrowSession`] of its execution context. Every other
//! conversion goes through the [`ArrowSession`] itself: importing Arrow schemas, fields, and data
//! types into Vortex ([`ArrowSession::from_arrow_schema`], [`ArrowSession::from_arrow_field`],
//! [`ArrowSession::from_arrow_datatype`]), importing Arrow arrays and record batches
//! ([`ArrowSession::from_arrow_array`], [`ArrowSession::from_arrow_record_batch`]), and exporting
//! Vortex dtypes to Arrow ([`ArrowSession::to_arrow_schema`], [`ArrowSession::to_arrow_field`],
//! [`ArrowSession::to_arrow_datatype`]).
//!
//! The [`ArrowSession`] session variable is created on first use of [`ArrowSessionExt::arrow`], so
//! no explicit registration is required. It can look up the extension types of its Vortex session
//! by their Vortex IDs (see [`ArrowSession`]).

pub mod convert;
mod datum;
pub mod dtype;
mod executor;

mod extension;
pub use extension::ArrowExtVTable;
pub use extension::ArrowMetadataPolicy;

pub mod ext_metadata;
mod iter;
mod null_buffer;
mod options;
mod run_end_import;
mod scalar;
mod session;
#[cfg(test)]
mod tests;

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

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Utilities and interface to convert DataFusion types to Vortex types.
//!
//! Currently includes:
//! [`ExpressionConverter`] - Controls the rewrite of DataFusion expressions to Vortex expressions, and whether they can
//! be pushed into the underlying scan. A default implementation is provided.
//! [`FromDataFusion`] - Converts a DataFusion type into a Vortex type infallible.
//! [TryToDataFusion] - Fallibly converts a Vortex type to a DataFusion type.

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::Field;
use vortex::array::ArrayRef;
use vortex::array::VortexSessionExecute;
use vortex::error::VortexResult;
use vortex::session::VortexSession;
use vortex_arrow::ExecuteArrowExt;

pub(crate) mod exprs;
mod scalars;
pub(crate) mod schema;
pub(crate) mod stats;

pub use exprs::DefaultExpressionConverter;
pub use exprs::ExpressionConverter;
pub use exprs::ProcessedProjection;
pub use exprs::convertible::*;
pub use scalars::scalar_from_df;

/// Convert a scan chunk using the non-nullable struct field for its projected Arrow columns.
pub(crate) fn to_record_batch(
    chunk: ArrayRef,
    target_field: &Field,
    session: &VortexSession,
) -> VortexResult<RecordBatch> {
    let arrow = chunk.execute_arrow(Some(target_field), &mut session.create_execution_ctx())?;
    Ok(RecordBatch::from(arrow.as_struct().clone()))
}

/// First-party trait for implementing conversion from DataFusion types to Vortex types.
pub trait FromDataFusion<D: ?Sized>: Sized {
    /// Convert to this Vortex type from the input DataFusion type.
    fn from_df(df: &D) -> Self;
}

/// First-party trait for implementing fallible conversions from Vortex to DataFusion types.
pub trait TryToDataFusion<D> {
    /// Try to convert this Vortex type from the input DataFusion type.
    fn try_to_df(&self) -> VortexResult<D>;
}

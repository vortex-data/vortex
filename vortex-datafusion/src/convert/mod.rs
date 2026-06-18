// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Utilities and interface to convert DataFusion types to Vortex types.
//!
//! Currently includes:
//! [`ExpressionConvertor`] - Controls the rewrite of DataFusion expressions to Vortex expressions, and whether they can
//! be pushed into the underlying scan. A default implementation is provided.
//! [`FromDataFusion`] - Converts a DataFusion type into a Vortex type infallible.
//! [TryToDataFusion] - Fallibly converts a Vortex type to a DataFusion type.

use vortex::error::VortexResult;

pub(crate) mod exprs;
mod scalars;
pub(crate) mod schema;
pub(crate) mod stats;

pub use exprs::DefaultExpressionConvertor;
pub use exprs::ExpressionConvertor;

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

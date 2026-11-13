// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversions between DataFusion and Vortex: physical expressions, scalars, schemas and
//! statistics.
//!
//! This crate depends on `vortex-array` and `vortex-arrow` but not on the Vortex file format, so
//! a DataFusion host that sends Vortex expressions to another process can link it alone.
//! `vortex-datafusion` re-exports it as `vortex_datafusion::convert`.
//!
//! Currently includes:
//! [`ExpressionConverter`] - Controls the rewrite of DataFusion expressions to Vortex expressions, and whether they can
//! be pushed into the underlying scan. A default implementation is provided.
//! [`FromDataFusion`] - Converts a DataFusion type into a Vortex type infallible.
//! [TryToDataFusion] - Fallibly converts a Vortex type to a DataFusion type.

use std::fmt::Debug;

use datafusion_common::stats::Precision as DFPrecision;
use vortex_array::expr::stats::Precision;
use vortex_error::VortexResult;

pub mod exprs;
mod scalars;
pub mod schema;
pub mod stats;

pub use exprs::DefaultExpressionConverter;
pub use exprs::ExpressionConverter;
pub use exprs::ProcessedProjection;
pub use scalars::scalar_from_df;

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

/// Extension trait to convert Vortex's [`Precision`] to DataFusion's [`DFPrecision`].
pub trait PrecisionExt<T>
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    /// Convert `Precision` to the DataFusion equivalent.
    fn to_df(self) -> DFPrecision<T>;
}

impl<T> PrecisionExt<T> for Precision<T>
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    fn to_df(self) -> DFPrecision<T> {
        match self {
            Precision::Exact(v) => DFPrecision::Exact(v),
            Precision::Inexact(v) => DFPrecision::Inexact(v),
            Precision::Absent => DFPrecision::Absent,
        }
    }
}

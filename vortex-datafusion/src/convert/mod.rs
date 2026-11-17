// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Utilities and interface to convert DataFusion types to Vortex types.
//!
//! Currently includes:
//! [`ExpressionConverter`] - Controls the rewrite of DataFusion expressions to Vortex expressions, and whether they can
//! be pushed into the underlying scan. A default implementation is provided.
//! [`FromDataFusion`] - Converts a DataFusion type into a Vortex type infallible.
//! [TryToDataFusion] - Fallibly converts a Vortex type to a DataFusion type.

use vortex::error::VortexResult;

pub(crate) mod exprs;
mod scalars;
pub(crate) mod schema;
pub(crate) mod stats;

pub use exprs::Convertible;
pub use exprs::ConvertibleCtx;
pub use exprs::ConvertibleFixedArity;
pub use exprs::DefaultExpressionConverter;
pub use exprs::ExpressionConverter;
pub use exprs::ProcessedProjection;
// pub use exprs::convertible::BinaryConversion;
// pub use exprs::convertible::CaseConversion;
// pub use exprs::convertible::ConvertArrayLength;
// pub use exprs::convertible::ConvertArraySum;
// pub use exprs::convertible::ConvertBinary;
// pub use exprs::convertible::ConvertCase;
// pub use exprs::convertible::ConvertCast;
// pub use exprs::convertible::ConvertColumn;
// pub use exprs::convertible::ConvertContains;
// pub use exprs::convertible::ConvertEndsWith;
// pub use exprs::convertible::ConvertGetField;
// pub use exprs::convertible::ConvertInList;
// pub use exprs::convertible::ConvertIsNotNull;
// pub use exprs::convertible::ConvertIsNull;
// pub use exprs::convertible::ConvertLambdaVariable;
// pub use exprs::convertible::ConvertLike;
// pub use exprs::convertible::ConvertLiteral;
// pub use exprs::convertible::ConvertLower;
// pub use exprs::convertible::ConvertNot;
// pub use exprs::convertible::ConvertOctetLength;
// pub use exprs::convertible::ConvertStartsWith;
// pub use exprs::convertible::ConvertStrpos;
// pub use exprs::convertible::InListConversion;
// pub use exprs::try_convert_node;
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

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar function expressions defined on tensor and tensor-like extension types.

use std::fmt;

pub mod cosine_similarity;
pub mod inner_product;
pub mod l2_norm;

/// Options for tensor-related expressions that might have error.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub enum ApproxOptions {
    /// Computes the exact result.
    #[default]
    Exact,
    /// Allows approximate results.
    Approximate,
}

impl ApproxOptions {
    /// Returns `true` if the option is [`Exact`](Self::Exact).
    pub fn is_exact(&self) -> bool {
        matches!(self, Self::Exact)
    }

    /// Returns `true` if the option is [`Approximate`](Self::Approximate).
    pub fn is_approx(&self) -> bool {
        matches!(self, Self::Approximate)
    }
}

impl fmt::Display for ApproxOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact => write!(f, "Exact"),
            Self::Approximate => write!(f, "Approximate"),
        }
    }
}

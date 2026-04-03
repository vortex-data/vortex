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
    #[default]
    Exact,
    Approximate,
}

impl fmt::Display for ApproxOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact => write!(f, "Exact"),
            Self::Approximate => write!(f, "Approximate"),
        }
    }
}

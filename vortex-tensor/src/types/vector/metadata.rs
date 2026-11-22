// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Metadata that declares the element type and dimension of a [`Vector`](super::Vector).
//!
//! The serialized metadata is empty, since the stored storage dtype carries both facts on disk.

use std::fmt;

use vortex_array::dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

/// The metadata of the [`Vector`](super::Vector) type: its element type and number of dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorMetadata {
    /// The floating-point element type (`f16`, `f32`, or `f64`). Elements are non-nullable.
    element_ptype: PType,
    /// The number of dimensions of the vector. This is always fixed.
    dimensions: u32,
}

impl VectorMetadata {
    /// Creates metadata for vectors with the given element type and dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if the element type is not a float.
    pub fn try_new(element_ptype: PType, dimensions: u32) -> VortexResult<Self> {
        vortex_ensure!(
            element_ptype.is_float(),
            "Vector element type must be a float, got {element_ptype}"
        );

        Ok(Self {
            element_ptype,
            dimensions,
        })
    }

    /// Returns the element type of the vectors.
    pub fn element_ptype(&self) -> PType {
        self.element_ptype
    }

    /// Returns the number of dimensions of the vector.
    pub fn dimensions(&self) -> u32 {
        self.dimensions
    }
}

/// Omits metadata text because the extension dtype's display includes the element type and
/// dimension in its storage dtype.
impl fmt::Display for VectorMetadata {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

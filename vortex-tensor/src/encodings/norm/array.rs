// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;

/// A normalized array that stores unit-normalized vectors alongside their original L2 norms.
///
/// Each vector in the array is divided by its L2 norm, producing a unit-normalized vector. The
/// original norms are stored separately so that the original vectors can be reconstructed.
#[derive(Debug, Clone)]
pub struct NormVectorArray {
    /// The backing vector array that has been unit normalized.
    ///
    /// The underlying elements of the vector array must be floating-point.
    vector_array: ArrayRef,

    /// The L2 (Frobenius) norms of each vector.
    ///
    /// This must have the same dtype as the elements of the vector array.
    norms: ArrayRef,
}

impl NormVectorArray {
    /// Returns a reference to the backing vector array that has been unit normalized.
    pub fn vector_array(&self) -> &ArrayRef {
        &self.vector_array
    }

    /// Returns a reference to the L2 (Frobenius) norms of each vector.
    pub fn norms(&self) -> &ArrayRef {
        &self.norms
    }
}

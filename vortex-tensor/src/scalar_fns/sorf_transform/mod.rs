// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SORF (Structured Orthogonal Random Features) inverse-rotation scalar function.
//!
//! Wraps an FSL child (e.g. `FSL(Dict(codes, centroids))`) and applies the inverse
//! Walsh-Hadamard-based structured rotation at execution time, producing a [`Vector`]
//! extension array with the original (pre-padding) dimensionality.
//!
//! The rotation parameters are stored as a deterministic PRNG seed in [`SorfOptions`],
//! so the `RotationMatrix` is reconstructed cheaply at decode time.
//!
//! [`Vector`]: crate::vector::Vector

use std::fmt;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::ScalarFn;
use vortex_error::VortexResult;

mod rotation;
pub use rotation::RotationMatrix;

mod vtable;

/// Inverse SORF rotation scalar function.
///
/// Applies the inverse structured Walsh-Hadamard rotation to an FSL child, truncates
/// from padded dimension to the original dimension, casts to the target element type,
/// and wraps in a [`Vector`] extension array.
#[derive(Clone)]
pub struct SorfTransform;

/// Options for the SORF inverse-rotation scalar function.
///
/// Stored in the [`ScalarFnArray`] and used to deterministically reconstruct the
/// `RotationMatrix` at decode time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SorfOptions {
    /// PRNG seed used to generate the random sign diagonals.
    pub seed: u64,
    /// Number of sign-diagonal + WHT rounds in the structured rotation.
    pub num_rounds: u8,
    /// Original vector dimension (before power-of-2 padding).
    pub dimension: u32,
    /// Target output element type (e.g. `F16`, `F32`, `F64`).
    pub element_ptype: PType,
}

impl SorfTransform {
    /// Creates a new [`ScalarFn`] wrapping the SORF inverse rotation with the given options.
    pub fn new(options: &SorfOptions) -> ScalarFn<SorfTransform> {
        ScalarFn::new(SorfTransform, options.clone())
    }

    /// Constructs a validated [`ScalarFnArray`] that lazily applies inverse SORF rotation.
    ///
    /// The `child` must be a `FixedSizeList` (or array that executes to one) with
    /// `list_size == padded_dim` (i.e. `dimension.next_power_of_two()`).
    pub fn try_new_array(
        options: &SorfOptions,
        child: ArrayRef,
        len: usize,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(SorfTransform::new(options).erased(), vec![child], len)
    }
}

impl fmt::Display for SorfOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SorfOptions(seed={}, rounds={}, dim={}, ptype={})",
            self.seed, self.num_rounds, self.dimension, self.element_ptype
        )
    }
}

#[cfg(test)]
mod tests;

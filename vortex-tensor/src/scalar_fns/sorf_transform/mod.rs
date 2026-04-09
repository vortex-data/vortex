// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SORF inverse transform scalar function.
//!
//! SORF (Structured Orthogonal Random Features, [Yu et al. 2016][sorf-paper]) is a fast structured
//! approximation to a random orthogonal matrix. It composes random sign diagonals with the
//! Walsh-Hadamard transform to achieve O(d log d) matrix-vector products instead of the O(d^2) cost
//! of a dense orthogonal matrix.
//!
//! This module wraps an FSL child (e.g. `FSL(Dict(codes, centroids))`) and applies the inverse SORF
//! transform at execution time, producing a [`Vector`] extension array with the original
//! (pre-padding) dimensionality.
//!
//! The transform parameters are stored as a deterministic PRNG seed in [`SorfOptions`], so the
//! [`SorfMatrix`] is reconstructed cheaply at decode time.
//!
//! **All SORF computation happens in f32.** Input elements of other float types (f16, f64) are cast
//! to f32 before the transform, and the result is cast back to the target type specified by
//! [`SorfOptions::element_ptype`].
//!
//! [sorf-paper]: https://proceedings.neurips.cc/paper_files/paper/2016/file/53adaf494dc89ef7196d73636eb2451b-Paper.pdf
//! [`Vector`]: crate::vector::Vector

use std::fmt;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::ScalarFn;
use vortex_error::VortexResult;

mod rotation;
pub use rotation::SorfMatrix;

mod vtable;

/// Inverse SORF orthogonal transform scalar function.
///
/// Applies the inverse structured Walsh-Hadamard orthogonal transform to an FSL child,
/// truncates from padded dimension to the original dimension, casts to the target element
/// type, and wraps in a [`Vector`] extension array.
#[derive(Clone)]
pub struct SorfTransform;

/// Options for the SORF inverse transform scalar function.
///
/// Stored in the [`ScalarFnArray`] and used to deterministically reconstruct the
/// [`SorfMatrix`] at decode time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SorfOptions {
    /// PRNG seed used to generate the random sign diagonals.
    pub seed: u64,
    /// Number of sign-diagonal + WHT rounds in the structured orthogonal transform.
    pub num_rounds: u8,
    /// Original vector dimension (before power-of-2 padding).
    pub dimension: u32,
    /// Target output element type (e.g. `F16`, `F32`, `F64`).
    pub element_ptype: PType,
}

impl SorfTransform {
    /// Creates a new [`ScalarFn`] wrapping the SORF inverse transform with the given options.
    pub fn new(options: &SorfOptions) -> ScalarFn<SorfTransform> {
        ScalarFn::new(SorfTransform, options.clone())
    }

    /// Constructs a validated [`ScalarFnArray`] that lazily applies the inverse SORF transform.
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

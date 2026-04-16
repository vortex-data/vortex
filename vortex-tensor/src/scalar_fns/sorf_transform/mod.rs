// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SORF inverse transform scalar function.
//!
//! SORF (Structured Orthogonal Random Features, [Yu et al. 2016][sorf-paper]) is a fast structured
//! approximation to a random orthogonal matrix. It composes random sign diagonals with the
//! Walsh-Hadamard transform to achieve O(d log d) matrix-vector products instead of the O(d^2) cost
//! of a dense orthogonal matrix.
//!
//! This module wraps a [`Vector`] extension array whose dimension is the padded SORF dimension
//! (e.g. a `Vector` wrapping `FSL(Dict(codes, centroids))`) and applies the inverse SORF transform
//! at execution time, producing a [`Vector`] extension array with the original (pre-padding)
//! dimensionality.
//!
//! The transform parameters are stored as a deterministic seed in [`SorfOptions`], so the
//! [`SorfMatrix`] is reconstructed cheaply at decode time. Sign diagonals are defined by Vortex's
//! frozen local SplitMix64 stream contract rather than by an external RNG crate.
//!
//! # Input element type: `f32` only (TODO(connor): for now...)
//!
//! The child [`Vector`] **must** have `f32` storage elements. This is a hard constraint that is
//! enforced by `SorfTransform`'s `return_dtype` check. Callers with `f16` or `f64` source data need
//! to cast to `f32` before wrapping in a [`Vector`] and handing it to SorfTransform.
//!
//! The reason for this constraint is that TurboQuant (the only production caller today) stores its
//! dictionary centroids as `f32`, and the SORF transform itself operates internally in `f32`.
//!
//! Supporting other float storage types would require an implicit up-/down-cast that we do not yet
//! want to bake into SorfTransform. This restriction is intentional and may be relaxed in the
//! future, but today it is load-bearing.
//!
//! # Output element type
//!
//! The output [`Vector`]'s element type is whatever [`SorfOptions::element_ptype`] is set to. It
//! does **not** have to match the child's `f32` storage: we apply an explicit `f32 -> T` cast
//! while materializing the output. This lets SorfTransform hand its result directly to a
//! downstream consumer (e.g. [`L2Denorm`](crate::scalar_fns::l2_denorm::L2Denorm)) whose
//! element-type expectation may differ from the `f32` the transform operated on internally.
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
use vortex_error::vortex_ensure;

mod rotation;
pub(crate) mod splitmix64;
pub use rotation::SorfMatrix;

mod vtable;

/// Inverse SORF orthogonal transform scalar function.
///
/// Takes a [`Vector`](crate::vector::Vector) extension child at the padded dimension with `f32`
/// storage, applies the inverse structured Walsh-Hadamard orthogonal transform, truncates to the
/// original (pre-padding) dimension, casts element-wise to [`SorfOptions::element_ptype`], and
/// wraps the result in a new [`Vector`](crate::vector::Vector) extension array.
///
/// See the [module-level docs](crate::scalar_fns::sorf_transform) for the rationale behind the
/// `f32`-only input constraint.
#[derive(Clone)]
pub struct SorfTransform;

/// Options for the SORF inverse transform scalar function.
///
/// Stored in the [`ScalarFnArray`] and used to deterministically reconstruct the
/// [`SorfMatrix`] at decode time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SorfOptions {
    /// Seed used to generate the structured sign diagonals via Vortex's frozen SplitMix64 stream.
    pub seed: u64,
    /// Number of sign-diagonal + WHT rounds in the structured orthogonal transform.
    pub num_rounds: u8,
    /// Original vector dimension (before power-of-2 padding). The output
    /// [`Vector`](crate::vector::Vector) has this dimension.
    pub dimension: u32,
    /// Element type of the output [`Vector`](crate::vector::Vector). The child input must always
    /// be `f32`, but the output can be any float type (`F16`, `F32`, `F64`); the final
    /// `f32 -> element_ptype` cast happens while building the output.
    pub element_ptype: PType,
}

impl SorfTransform {
    /// Creates a new [`ScalarFn`] wrapping the SORF inverse transform with the given options.
    pub fn new(options: &SorfOptions) -> ScalarFn<SorfTransform> {
        ScalarFn::new(SorfTransform, options.clone())
    }

    /// Constructs a validated [`ScalarFnArray`] that lazily applies the inverse SORF transform.
    ///
    /// The `child` must be a [`Vector`] extension array (or an array that executes to one) with:
    ///
    /// - dimension equal to `padded_dim` (i.e. `options.dimension.next_power_of_two()`), and
    /// - `f32` storage elements. This is a hard requirement today; see the
    ///   [module-level docs](crate::scalar_fns::sorf_transform) for the rationale.
    ///
    /// The output [`Vector`] has dimension `options.dimension` and element type
    /// `options.element_ptype`.
    ///
    /// [`Vector`]: crate::vector::Vector
    pub fn try_new_array(
        options: &SorfOptions,
        child: ArrayRef,
        len: usize,
    ) -> VortexResult<ScalarFnArray> {
        validate_sorf_options(options)?;

        ScalarFnArray::try_new(SorfTransform::new(options).erased(), vec![child], len)
    }
}

/// Checks that the SORF configuration is valid.
pub(crate) fn validate_sorf_options(options: &SorfOptions) -> VortexResult<()> {
    vortex_ensure!(
        options.num_rounds >= 1,
        "SorfTransform num_rounds must be >= 1, got {}",
        options.num_rounds
    );
    vortex_ensure!(
        options.element_ptype.is_float(),
        "SorfTransform element_ptype must be a float type, got {}",
        options.element_ptype
    );
    Ok(())
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

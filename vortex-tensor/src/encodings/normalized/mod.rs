// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Normalized`] encoding: a norm-split physical layout for tensor-like columns.
//!
//! An [`Normalized`] array stores a tensor or vector column as two children:
//!
//! - `normalized`, a tensor-like column whose valid rows are unit-norm (or zero), and
//! - `norms`, a primitive float column holding the authoritative L2 norm of each row.
//!
//! The logical value of row `i` is `normalized[i] * norms[i]`, so canonicalizing the array
//! reconstructs the original tensor column. Splitting magnitude away from direction is what makes
//! the coordinates cheap to compress further: a unit-norm child has a bounded, well-conditioned
//! value range, and quantizing it only perturbs direction while the exact magnitude survives in
//! `norms`.
//!
//! Because the split is physical rather than logical, [`L2Norm`], [`InnerProduct`], and
//! [`CosineSimilarity`] can read straight through it instead of decoding first.
//!
//! [`L2Norm`]: crate::scalar_fns::l2_norm::L2Norm
//! [`InnerProduct`]: crate::scalar_fns::inner_product::InnerProduct
//! [`CosineSimilarity`]: crate::scalar_fns::cosine_similarity::CosineSimilarity

mod array;
pub use array::Normalized;
pub use array::NormalizedArray;
pub use array::NormalizedArraySlotsExt;
pub use array::NormalizedMetadata;
pub use array::NormalizedSlots;

mod compress;
pub use compress::NormalizedScheme;
pub use compress::normalize;
pub(crate) use compress::try_build_constant_normalized;

mod execute;

mod orientation;
pub(crate) use orientation::NormalizedOrientation;

mod rules;

mod validate;
pub use validate::validate_l2_normalized_rows_against_norms;

#[cfg(test)]
mod tests;

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Normalized`] encoding stores tensor directions separately from their L2 norms.
//!
//! [`Normalized`] defines the physical layout and its invariants. Use [`normalize`] to create an
//! exact split. [`L2Norm`], [`InnerProduct`], and [`CosineSimilarity`] can operate on the split
//! without decoding it first.
//!
//! [`normalize`]: crate::encodings::normalized::normalize
//! [`L2Norm`]: crate::scalar_fns::l2_norm::L2Norm
//! [`InnerProduct`]: crate::scalar_fns::inner_product::InnerProduct
//! [`CosineSimilarity`]: crate::scalar_fns::cosine_similarity::CosineSimilarity

mod array;
pub use array::Normalized;
pub use array::NormalizedArray;
pub use array::NormalizedArraySlotsExt;
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
pub use validate::validate_normalized_rows;

#[cfg(test)]
mod tests;

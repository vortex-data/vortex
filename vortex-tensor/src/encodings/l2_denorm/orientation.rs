// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;

use crate::encodings::l2_denorm::L2Denorm;

/// Classification of a binary operand pair by which side (if any) is [`L2Denorm`]-encoded.
///
/// Symmetric binary tensor operators ([`CosineSimilarity`], [`InnerProduct`]) have identical fast
/// paths for "only the lhs is denormalized" and "only the rhs is denormalized", plus a separate
/// fast path for "both are denormalized". Rather than hand-rolling the commutative swap at every
/// call site, callers classify their operands with [`Self::classify`] and match on the result.
///
/// [`CosineSimilarity`]: crate::scalar_fns::cosine_similarity::CosineSimilarity
/// [`InnerProduct`]: crate::scalar_fns::inner_product::InnerProduct
pub(crate) enum DenormOrientation<'a> {
    /// Both operands are [`L2Denorm`] arrays.
    Both {
        /// The left-hand operand.
        lhs: &'a ArrayRef,
        /// The right-hand operand.
        rhs: &'a ArrayRef,
    },

    /// Exactly one operand is an [`L2Denorm`] array; the other is a plain tensor column.
    One {
        /// The [`L2Denorm`]-encoded operand, whichever side it came from.
        denorm: &'a ArrayRef,
        /// The other operand.
        plain: &'a ArrayRef,
    },

    /// Neither operand is an [`L2Denorm`] array.
    Neither,
}

impl<'a> DenormOrientation<'a> {
    /// Classify `(lhs, rhs)` by which side (if any) is [`L2Denorm`]-encoded.
    pub(crate) fn classify(lhs: &'a ArrayRef, rhs: &'a ArrayRef) -> Self {
        match (lhs.is::<L2Denorm>(), rhs.is::<L2Denorm>()) {
            (true, true) => Self::Both { lhs, rhs },
            (true, false) => Self::One {
                denorm: lhs,
                plain: rhs,
            },
            (false, true) => Self::One {
                denorm: rhs,
                plain: lhs,
            },
            (false, false) => Self::Neither,
        }
    }
}

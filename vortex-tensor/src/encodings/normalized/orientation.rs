// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;

use crate::encodings::normalized::Normalized;

/// Classification of a binary operand pair by which side (if any) is [`Normalized`]-encoded.
///
/// Symmetric binary tensor operators ([`CosineSimilarity`], [`InnerProduct`]) have identical fast
/// paths when only one operand is [`Normalized`], plus a separate fast path when both operands are
/// [`Normalized`]. Rather than hand-rolling the commutative swap at every call site, callers
/// classify their operands with [`Self::classify`] and match on the result.
///
/// [`CosineSimilarity`]: crate::scalar_fns::cosine_similarity::CosineSimilarity
/// [`InnerProduct`]: crate::scalar_fns::inner_product::InnerProduct
pub(crate) enum NormalizedOrientation<'a> {
    /// Both operands are [`Normalized`] arrays.
    Both {
        /// The left-hand operand.
        lhs: &'a ArrayRef,
        /// The right-hand operand.
        rhs: &'a ArrayRef,
    },

    /// Exactly one operand is a [`Normalized`] array; the other is a plain tensor column.
    One {
        /// The [`Normalized`]-encoded operand, whichever side it came from.
        normalized_array: &'a ArrayRef,
        /// The other operand.
        plain: &'a ArrayRef,
    },

    /// Neither operand is a [`Normalized`] array.
    Neither,
}

impl<'a> NormalizedOrientation<'a> {
    /// Classify `(lhs, rhs)` by which side (if any) is [`Normalized`]-encoded.
    pub(crate) fn classify(lhs: &'a ArrayRef, rhs: &'a ArrayRef) -> Self {
        match (lhs.is::<Normalized>(), rhs.is::<Normalized>()) {
            (true, true) => Self::Both { lhs, rhs },
            (true, false) => Self::One {
                normalized_array: lhs,
                plain: rhs,
            },
            (false, true) => Self::One {
                normalized_array: rhs,
                plain: lhs,
            },
            (false, false) => Self::Neither,
        }
    }
}

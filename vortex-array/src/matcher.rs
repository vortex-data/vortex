// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;

/// Trait for matching array types.
pub trait Matcher {
    type Match<'a>;

    /// Whether this matcher is [`AnyCanonical`](crate::AnyCanonical), which the executor checks
    /// every iteration as its universal stop condition on top of the target matcher.
    ///
    /// Executing to canonical form - by far the common case - makes those two the same predicate,
    /// and this lets the executor answer both with one scan instead of repeating it per iteration.
    /// Only `AnyCanonical` may override it.
    const IS_ANY_CANONICAL: bool = false;

    /// Check if the given array matches this matcher type
    #[inline]
    fn matches(array: &ArrayRef) -> bool {
        Self::try_match(array).is_some()
    }

    /// Try to match the given array, returning the matched view type if successful.
    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>>;
}

/// Matches any array type (wildcard matcher)
#[derive(Debug)]
pub struct AnyArray;

impl Matcher for AnyArray {
    type Match<'a> = &'a ArrayRef;

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn matches(_array: &ArrayRef) -> bool {
        true
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        Some(array)
    }
}

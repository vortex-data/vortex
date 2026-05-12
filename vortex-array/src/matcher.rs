// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;

/// Trait for matching array types by reference.
pub trait Matcher {
    type Match<'a>;

    /// Check if the given array matches this matcher type.
    fn matches(array: &ArrayRef) -> bool {
        Self::try_match(array).is_some()
    }

    /// Try to match the given array, returning the matched view type if successful.
    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>>;
}

/// Trait for matching array types by owned value.
pub trait OwnedMatcher: Matcher {
    type OwnedMatch;

    /// Try to match the given array, returning the owned matched type if successful.
    fn maybe_match(array: ArrayRef) -> Option<Self::OwnedMatch>;
}

/// Matches any array type (wildcard matcher)
#[derive(Debug)]
pub struct AnyArray;

impl Matcher for AnyArray {
    type Match<'a> = &'a ArrayRef;

    #[inline(always)]
    fn matches(_array: &ArrayRef) -> bool {
        true
    }

    #[inline(always)]
    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        Some(array)
    }
}

impl OwnedMatcher for AnyArray {
    type OwnedMatch = ArrayRef;

    #[inline(always)]
    fn maybe_match(array: ArrayRef) -> Option<Self::OwnedMatch> {
        Some(array)
    }
}

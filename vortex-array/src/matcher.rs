// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Array;

/// Trait for matching array types.
pub trait Matcher {
    type Match<'a>;

    /// Check if the given array matches this matcher type
    fn matches(array: &dyn Array) -> bool {
        Self::try_match(array).is_some()
    }

    /// Try to match the given array, returning the matched view type if successful
    fn try_match<'a>(array: &'a dyn Array) -> Option<Self::Match<'a>>;
}

/// Matches any array type (wildcard matcher)
#[derive(Debug)]
pub struct AnyArray;

impl Matcher for AnyArray {
    type Match<'a> = &'a dyn Array;

    #[inline(always)]
    fn matches(_array: &dyn Array) -> bool {
        true
    }

    #[inline(always)]
    fn try_match(array: &dyn Array) -> Option<Self::Match<'_>> {
        Some(array)
    }
}

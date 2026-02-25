// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Array;

/// A super trait that allows us to define a generic associated type on `Matcher`.
///
/// We need to do this in this manner because of rust needs to keep forward compatibility.
/// See [this](https://github.com/rust-lang/rust/issues/87479) GitHub issue for more information.
pub trait MatcherType<Over: ?Sized> {
    /// The matched view type.
    type Match<'a>;
}

/// A trait for matching types.
pub trait Matcher<Over: ?Sized>: MatcherType<Over> {
    /// Check if the given item matches this matcher.
    fn matches(item: &Over) -> bool {
        Self::try_match(item).is_some()
    }

    /// Try to match the given item, returning the matched type if successful.
    fn try_match<'a>(item: &'a Over) -> Option<Self::Match<'a>>;
}

/// Matches any array type (wildcard matcher)
#[derive(Debug)]
pub struct AnyArray;

impl MatcherType<dyn Array> for AnyArray {
    type Match<'a> = &'a dyn Array;
}

impl Matcher<dyn Array> for AnyArray {
    #[inline(always)]
    fn matches(_array: &dyn Array) -> bool {
        true
    }

    #[inline(always)]
    fn try_match(array: &dyn Array) -> Option<Self::Match<'_>> {
        Some(array)
    }
}

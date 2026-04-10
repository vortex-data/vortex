// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;

/// Category bit for canonical arrays (Null, Bool, Primitive, Decimal, Struct, ListView,
/// FixedSizeList, VarBinView, Variant, Extension).
pub const CATEGORY_CANONICAL: u32 = 1 << 0;

/// Category bit for the Constant array encoding.
pub const CATEGORY_CONSTANT: u32 = 1 << 1;

/// Category bit for ScalarFn arrays (any scalar function expression).
pub const CATEGORY_SCALAR_FN: u32 = 1 << 2;

/// A precomputed hint describing what an outer array kind a matcher can accept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatcherHint {
    /// Matches exactly one encoding identified by its interned string index.
    ///
    /// The index is obtained via [`crate::intern`] on the encoding's or scalar
    /// function's ID string.
    Exact(u32),

    /// Matches any encoding whose category flags satisfy `(flags & mask) != 0`.
    ///
    /// Use the `CATEGORY_*` constants as the mask value.
    Category(u32),
}

/// Trait for matching array types.
pub trait Matcher {
    type Match<'a>;

    /// Returns a hint for the outer array kind this matcher may accept.
    ///
    /// Matchers that can only ever match a specific array encoding or category
    /// should override this so dispatchers can skip impossible rules before
    /// calling [`matches`](Self::matches).
    fn dispatch_hint() -> Option<MatcherHint> {
        None
    }

    /// Check if the given array matches this matcher type
    fn matches(array: &ArrayRef) -> bool {
        Self::try_match(array).is_some()
    }

    /// Try to match the given array, returning the matched view type if successful.
    fn try_match<'a>(array: &'a ArrayRef) -> Option<Self::Match<'a>>;
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

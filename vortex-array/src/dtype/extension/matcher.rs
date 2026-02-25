// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtVTable;
use crate::dtype::extension::typed::ExtDTypeInner;

/// A super trait that allows us to define a generic associated type on `Matcher`.
///
/// We need to have this because of https://github.com/rust-lang/rust/issues/87479
pub trait MatcherType {
    /// The matched view type.
    type Match<'a>;
}

/// A trait for matching extension dtypes.
pub trait Matcher<Over>: MatcherType {
    /// Check if the given extension dtype matches this matcher.
    fn matches(item: &Over) -> bool {
        Self::try_match(item).is_some()
    }

    /// Try to match the given extension type, returning the matched dtype if successful.
    fn try_match<'a>(item: &'a Over) -> Option<Self::Match<'a>>;
}

impl<V: ExtVTable> MatcherType for V {
    type Match<'a> = &'a V::Metadata;
}

impl<V: ExtVTable> Matcher<ExtDTypeRef> for V {
    fn matches(item: &ExtDTypeRef) -> bool {
        item.0.as_any().is::<ExtDTypeInner<V>>()
    }

    fn try_match<'a>(item: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        item.0
            .as_any()
            .downcast_ref::<ExtDTypeInner<V>>()
            .map(|inner| &inner.metadata)
    }
}

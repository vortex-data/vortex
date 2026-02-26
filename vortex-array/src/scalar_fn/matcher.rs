// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::typed::ScalarFnInner;

/// A trait for matching scalar functions.
pub trait Matcher {
    /// The matched view type.
    type Match<'a>;

    /// Check if the given scalar function matches this matcher.
    fn matches(item: &ScalarFnRef) -> bool {
        Self::try_match(item).is_some()
    }

    /// Try to match the given scalar function, returning the matched options if successful.
    fn try_match<'a>(item: &'a ScalarFnRef) -> Option<Self::Match<'a>>;
}

impl<V: ScalarFnVTable> Matcher for V {
    type Match<'a> = &'a V::Options;

    fn matches(item: &ScalarFnRef) -> bool {
        item.0.as_any().is::<ScalarFnInner<V>>()
    }

    fn try_match<'a>(item: &'a ScalarFnRef) -> Option<Self::Match<'a>> {
        item.0
            .as_any()
            .downcast_ref::<ScalarFnInner<V>>()
            .map(|inner| &inner.options)
    }
}

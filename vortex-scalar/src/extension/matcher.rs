// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::extension::ExtScalarAdapter;
use crate::extension::ExtScalarRef;
use crate::extension::ExtScalarVTable;

/// A trait for matching extension scalars.
pub trait Matcher {
    /// The matched view type.
    type Match<'a>;

    /// Check if the given extension scalar matches this matcher.
    fn matches(item: &ExtScalarRef) -> bool {
        Self::try_match(item).is_some()
    }

    /// Check if the given extension scalar matches this matcher.
    fn try_match<'a>(item: &'a ExtScalarRef) -> Option<Self::Match<'a>>;
}

impl<V: ExtScalarVTable> Matcher for V {
    type Match<'a> = &'a V::Value;

    fn matches(item: &ExtScalarRef) -> bool {
        item.0.as_any().is::<ExtScalarAdapter<V>>()
    }

    fn try_match<'a>(item: &'a ExtScalarRef) -> Option<Self::Match<'a>> {
        item.0
            .as_any()
            .downcast_ref::<ExtScalarAdapter<V>>()
            .map(|adapter| &adapter.value)
    }
}

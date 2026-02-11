// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::extension::ExtDTypeAdapter;
use crate::extension::ExtDTypeRef;
use crate::extension::ExtDTypeVTable;

/// A trait for matching extension dtypes.
pub trait Matcher {
    /// The matched view type.
    type Match<'a>;

    /// Check if the given extension dtype matches this matcher.
    fn matches(item: &ExtDTypeRef) -> bool {
        Self::try_match_into(item).is_some()
    }

    /// Check if the given extension dtype matches this matcher.
    fn try_match_into<'a>(item: &'a ExtDTypeRef) -> Option<Self::Match<'a>>;
}

impl<V: ExtDTypeVTable> Matcher for V {
    type Match<'a> = &'a V::Metadata;

    fn matches(item: &ExtDTypeRef) -> bool {
        item.0.as_any().is::<ExtDTypeAdapter<V>>()
    }

    fn try_match_into<'a>(item: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        item.0
            .as_any()
            .downcast_ref::<ExtDTypeAdapter<V>>()
            .map(|adapter| &adapter.metadata)
    }
}

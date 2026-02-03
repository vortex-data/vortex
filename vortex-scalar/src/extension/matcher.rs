// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ScalarValue;
use crate::extension::ExtScalarVTable;
use crate::extension::ExtensionScalar;

/// A trait for matching extension scalars.
pub trait Matcher {
    /// The matched view type.
    type Match<'a>;

    /// Check if the given extension scalar matches this matcher.
    fn matches(item: &ExtensionScalar) -> bool {
        Self::try_match(item).is_some()
    }

    /// Check if the given extension scalar matches this matcher.
    fn try_match<'a>(item: &'a ExtensionScalar) -> Option<Self::Match<'a>>;
}

impl<V: ExtScalarVTable> Matcher for V {
    type Match<'a> = &'a ScalarValue;

    fn matches(item: &ExtensionScalar) -> bool {
        item.ext_dtype.is::<V>()
    }

    fn try_match<'a>(item: &'a ExtensionScalar) -> Option<Self::Match<'a>> {
        item.ext_dtype
            .is::<V>()
            .then(|| item.ext_scalar.map(|s| s.storage()))
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtVTable;

/// A trait for matching extension dtypes.
pub trait Matcher {
    /// The matched view type.
    type Match<'a>;

    /// Check if the given extension dtype matches this matcher.
    fn matches(item: &ExtDTypeRef) -> bool {
        Self::try_match(item).is_some()
    }

    /// Try to match the given extension type, returning the matched dtype if successful.
    fn try_match<'a>(item: &'a ExtDTypeRef) -> Option<Self::Match<'a>>;
}

impl<V: ExtVTable> Matcher for V {
    type Match<'a> = &'a V::Metadata;

    fn matches(ext_dtype: &ExtDTypeRef) -> bool {
        ext_dtype.0.as_any().is::<ExtDType<V>>()
    }

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        ext_dtype
            .0
            .as_any()
            .downcast_ref::<ExtDType<V>>()
            .map(|inner| inner.metadata())
    }
}

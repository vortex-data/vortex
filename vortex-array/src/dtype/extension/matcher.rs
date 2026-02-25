// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtVTable;
use crate::dtype::extension::typed::ExtDTypeInner;
use crate::matcher::Matcher;
use crate::matcher::MatcherType;

impl<V: ExtVTable> MatcherType<ExtDTypeRef> for V {
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

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

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
    type Match<'a> = V::Value<'a>;

    fn matches(item: &ExtensionScalar) -> bool {
        item.ext_dtype.is::<V>()
    }

    fn try_match<'a>(item: &'a ExtensionScalar) -> Option<Self::Match<'a>> {
        if let Some(metadata) = item.ext_dtype.metadata_opt::<V>() {
            item.ext_scalar.map(|s| {
                let vtable =
                    s.0.vtable_any()
                        .downcast_ref::<V>()
                        .vortex_expect("ExtScalarVTable downcast failed");
                vtable.unpack(
                    metadata,
                    item.ext_dtype.storage_dtype(),
                    item.ext_scalar.map(|s| s.storage()),
                )
            })
        } else {
            None
        }
    }
}

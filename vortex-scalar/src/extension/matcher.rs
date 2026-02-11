// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition of [`Matcher`] and a blanket implementation for [`ExtScalarVTable`].

use vortex_error::VortexExpect;

use crate::ExtScalar;
use crate::extension::ExtScalarVTable;

/// A trait for matching extension scalars.
pub trait Matcher {
    /// The matched view type.
    type Match<'a>;

    /// Check if the given extension scalar matches this matcher.
    fn matches(item: &ExtScalar) -> bool {
        Self::try_match(item).is_some()
    }

    /// Try to match the scalar into the correct view type.
    ///
    /// Returns `None` if unable to match to the type, or if the scalar is null.
    fn try_match<'a>(item: &'a ExtScalar) -> Option<Self::Match<'a>>;
}

impl<V: ExtScalarVTable> Matcher for V {
    type Match<'a> = Option<V::Value<'a>>;

    fn matches(item: &ExtScalar) -> bool {
        item.ext_dtype().is::<V>()
    }

    /// Returns `None` if unable to match with the vtable, `Some(None)` if scalar is null, and
    /// `Some(Some(_))` there is a matched value.
    fn try_match<'a>(item: &'a ExtScalar) -> Option<Option<V::Value<'a>>> {
        let metadata = item.ext_dtype().metadata_opt::<V>()?;

        let Some(ext_value_ref) = item.ext_value() else {
            // If there the value is null, we cannot match on it.
            return Some(None);
        };

        let vtable = ext_value_ref
            .try_get_vtable::<V>()
            .vortex_expect("we were able to match the dtype vtable but not the scalar value vtable")
            .clone();

        Some(Some(vtable.unpack(
            metadata,
            item.ext_dtype().storage_dtype(),
            ext_value_ref.storage_value(),
        )))
    }
}

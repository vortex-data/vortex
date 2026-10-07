// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;

/// Trait for matching array types.
pub trait Matcher {
    type Match<'a>;

    /// Check if the given array matches this matcher type
    #[inline]
    fn matches(array: &ArrayRef) -> bool {
        Self::try_match(array).is_some()
    }

    /// Try to match the given array, returning the matched view type if successful.
    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>>;
}

/// Matches any array type (wildcard matcher)
#[derive(Debug)]
pub struct AnyArray;

impl Matcher for AnyArray {
    type Match<'a> = &'a ArrayRef;

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn matches(_array: &ArrayRef) -> bool {
        true
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        Some(array)
    }
}

/// Defines a [`Matcher`] for a fixed set of array vtables and the enum of typed views it returns.
///
/// The concrete vtable's `TypeId` is read with one virtual call and compared against each member,
/// so this is cheaper than chaining `is::<A>() || is::<B>()`, which makes one call per member.
/// Each member adds a `TypeId` comparison; a large set on a hot path may warrant a dedicated
/// classification method on `DynArrayData` instead, as `AnyCanonical` uses.
macro_rules! vtable_set_matcher {
    (
        $(#[$matcher_meta:meta])*
        $matcher_vis:vis struct $matcher:ident;

        $(#[$view_meta:meta])*
        $view_vis:vis enum $view:ident { $($kind:ident($vtable:ty)),+ $(,)? }
    ) => {
        $(#[$matcher_meta])*
        $matcher_vis struct $matcher;

        $(#[$view_meta])*
        $view_vis enum $view<'a> {
            $($kind($crate::ArrayView<'a, $vtable>)),+
        }

        impl $crate::matcher::Matcher for $matcher {
            type Match<'a> = $view<'a>;

            #[inline]
            fn matches(array: &$crate::ArrayRef) -> bool {
                let id = array.vtable_type_id();
                $(id == ::std::any::TypeId::of::<$vtable>())||+
            }

            #[inline]
            fn try_match(array: &$crate::ArrayRef) -> Option<Self::Match<'_>> {
                let id = array.vtable_type_id();
                $(if id == ::std::any::TypeId::of::<$vtable>() {
                    // SAFETY: the concrete vtable's `TypeId` equals `$vtable`'s.
                    return Some($view::$kind(unsafe { array.as_typed_unchecked::<$vtable>() }));
                })+
                None
            }
        }
    };
}

pub(crate) use vtable_set_matcher;

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::array::new_foreign_array;
    use crate::arrays::Constant;
    use crate::arrays::ConstantArray;
    use crate::arrays::Null;
    use crate::arrays::NullArray;
    use crate::arrays::Primitive;

    vtable_set_matcher! {
        struct NullOrConstant;

        enum NullOrConstantView {
            Null(Null),
            Constant(Constant),
        }
    }

    #[test]
    fn vtable_set_matcher_matches_members() {
        let null = NullArray::new(3).into_array();
        assert!(null.is::<NullOrConstant>());
        match null.as_opt::<NullOrConstant>() {
            Some(NullOrConstantView::Null(n)) => assert_eq!(n.len(), 3),
            _ => panic!("expected a null view"),
        }

        let constant = ConstantArray::new(1i32, 3).into_array();
        assert!(constant.is::<NullOrConstant>());
        match constant.as_opt::<NullOrConstant>() {
            Some(NullOrConstantView::Constant(c)) => assert_eq!(c.len(), 3),
            _ => panic!("expected a constant view"),
        }
    }

    #[test]
    fn vtable_set_matcher_rejects_other_vtables() -> VortexResult<()> {
        let primitive = buffer![1i32, 2, 3].into_array();
        assert!(!primitive.is::<NullOrConstant>());
        assert!(primitive.as_opt::<NullOrConstant>().is_none());
        assert!(primitive.is::<Primitive>());

        let null = NullArray::new(3).into_array();
        let foreign: ArrayRef = new_foreign_array(
            null.encoding_id(),
            null.dtype().clone(),
            null.len(),
            vec![],
            vec![],
            Default::default(),
        )?;
        assert!(!foreign.is::<NullOrConstant>());
        assert!(!foreign.is::<Null>());
        Ok(())
    }
}

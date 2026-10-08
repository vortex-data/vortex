// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::TypeId;

use crate::ArrayRef;
use crate::array::ArrayId;
use crate::array::VTable;

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

/// Whether `array`, whose encoding ID is `id`, has the concrete vtable `V`.
///
/// A [`VTable::static_id`] that differs from `id` rejects without a virtual call, and a reserved
/// canonical or constant ID that equals it accepts without one, since construction enforces that
/// only its own vtable carries it. Otherwise the concrete vtable's `TypeId` decides, because other
/// encoding IDs are not guaranteed unique per vtable. `type_id` caches that virtual call across
/// several checks of the same array.
#[inline]
pub(crate) fn vtable_matches<V: VTable>(
    array: &ArrayRef,
    id: ArrayId,
    type_id: &mut Option<TypeId>,
) -> bool {
    if let Some(static_id) = V::static_id() {
        if id != static_id {
            return false;
        }
        if static_id.is_canonical_or_constant() {
            return true;
        }
    }
    *type_id.get_or_insert_with(|| array.vtable_type_id()) == TypeId::of::<V>()
}

/// Defines a [`Matcher`] for a fixed set of array vtables and the enum of typed views it returns.
///
/// Each member is checked with [`vtable_matches`], so members with a static encoding ID are
/// decided by the array's inline ID and at most one virtual call reads the concrete vtable's
/// `TypeId`.
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
                let id = array.encoding_id();
                let mut type_id = None;
                $($crate::matcher::vtable_matches::<$vtable>(array, id, &mut type_id))||+
            }

            #[inline]
            fn try_match(array: &$crate::ArrayRef) -> Option<Self::Match<'_>> {
                let id = array.encoding_id();
                let mut type_id = None;
                $(if $crate::matcher::vtable_matches::<$vtable>(array, id, &mut type_id) {
                    // SAFETY: `vtable_matches` established that the concrete vtable is `$vtable`.
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
    use std::fmt::Display;
    use std::fmt::Formatter;
    use std::hash::Hasher;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_panic;
    use vortex_session::VortexSession;
    use vortex_session::registry::Id;

    use crate::AnyCanonical;
    use crate::ArrayEq;
    use crate::ArrayHash;
    use crate::ArrayRef;
    use crate::EqMode;
    use crate::ExecutionCtx;
    use crate::ExecutionResult;
    use crate::IntoArray;
    use crate::array::Array;
    use crate::array::ArrayId;
    use crate::array::ArrayParts;
    use crate::array::ArrayView;
    use crate::array::VTable;
    use crate::array::new_foreign_array;
    use crate::array::vtable::NotSupported;
    use crate::array::vtable::ValidityVTable;
    use crate::array::vtable::with_empty_buffers;
    use crate::arrays::Constant;
    use crate::arrays::ConstantArray;
    use crate::arrays::Null;
    use crate::arrays::NullArray;
    use crate::arrays::Primitive;
    use crate::buffer::BufferHandle;
    use crate::dtype::DType;
    use crate::serde::ArrayChildren;
    use crate::validity::Validity;

    /// A vtable that reports whatever encoding ID it is given.
    #[derive(Clone, Debug)]
    struct Impostor(ArrayId);

    #[derive(Clone, Debug)]
    struct ImpostorData;

    impl Display for ImpostorData {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.write_str("impostor")
        }
    }

    impl ArrayHash for ImpostorData {
        fn array_hash<H: Hasher>(&self, _state: &mut H, _eq_mode: EqMode) {}
    }

    impl ArrayEq for ImpostorData {
        fn array_eq(&self, _other: &Self, _eq_mode: EqMode) -> bool {
            true
        }
    }

    impl ValidityVTable<Impostor> for Impostor {
        fn validity(_array: ArrayView<'_, Impostor>) -> VortexResult<Validity> {
            Ok(Validity::NonNullable)
        }
    }

    impl VTable for Impostor {
        type TypedArrayData = ImpostorData;
        type OperationsVTable = NotSupported;
        type ValidityVTable = Self;

        fn id(&self) -> ArrayId {
            self.0
        }

        fn validate(
            &self,
            _data: &Self::TypedArrayData,
            _dtype: &DType,
            _len: usize,
            _slots: &[Option<ArrayRef>],
        ) -> VortexResult<()> {
            Ok(())
        }

        fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
            0
        }

        fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
            vortex_panic!("Impostor buffer index {idx} out of bounds")
        }

        fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
            None
        }

        fn serialize(
            _array: ArrayView<'_, Self>,
            _session: &VortexSession,
        ) -> VortexResult<Option<Vec<u8>>> {
            Ok(None)
        }

        fn deserialize(
            &self,
            _dtype: &DType,
            _len: usize,
            _metadata: &[u8],
            _buffers: &[BufferHandle],
            _children: &dyn ArrayChildren,
            _session: &VortexSession,
        ) -> VortexResult<ArrayParts<Self>> {
            vortex_bail!("Impostor cannot be deserialized")
        }

        fn with_buffers(
            &self,
            array: ArrayView<'_, Self>,
            buffers: &[BufferHandle],
        ) -> VortexResult<ArrayParts<Self>> {
            with_empty_buffers(self, array, buffers)
        }

        fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
            vortex_panic!("Impostor slot index {idx} out of bounds")
        }

        fn execute(_array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
            vortex_bail!("Impostor cannot be executed")
        }
    }

    #[expect(clippy::disallowed_methods, reason = "the test needs a dynamic id")]
    fn impostor(id: &str) -> VortexResult<ArrayRef> {
        Ok(Array::try_from_parts(ArrayParts::new(
            Impostor(Id::new(id)),
            DType::Null,
            3,
            ImpostorData,
            Default::default(),
        ))?
        .into_array())
    }

    #[rstest]
    #[case::canonical("vortex.primitive")]
    #[case::constant("vortex.constant")]
    #[should_panic(expected = "is reserved for a built-in vtable")]
    fn reserved_id_from_another_vtable_panics(#[case] id: &str) {
        drop(impostor(id));
    }

    #[test]
    fn unreserved_id_matches_by_vtable() -> VortexResult<()> {
        let array = impostor("vortex.test.impostor")?;
        assert!(array.is::<Impostor>());
        assert!(!array.is::<Primitive>());
        assert!(!array.is::<AnyCanonical>());
        assert!(!array.is::<NullOrConstant>());
        Ok(())
    }

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

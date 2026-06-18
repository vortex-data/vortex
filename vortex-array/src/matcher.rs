// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::ArrayRef;
use crate::array::ArrayId;
use crate::array::ParentRef;
use crate::array::ParentView;
use crate::array::VTable;
use crate::dtype::DType;

mod private {
    pub trait Sealed {}
}

impl private::Sealed for ArrayRef {}
impl private::Sealed for ParentRef<'_> {}

/// A parent array that matchers can inspect without forcing materialization.
///
/// Implemented by [`ArrayRef`] (always heap-backed) and [`ParentRef`] (heap- or
/// stack-backed), so a single [`Matcher`] implementation serves both the execute
/// dispatch chain (heap arrays) and the parent-reduce dispatch chain
/// (stack-allocated construction parts).
///
/// This trait is sealed: matcher code can rely on materialization being free for
/// heap-backed parents and explicit for stack-backed ones.
pub trait AsParent: private::Sealed {
    /// Returns the encoding id of the parent.
    fn encoding_id(&self) -> ArrayId;

    /// Returns the dtype of the parent.
    fn dtype(&self) -> &DType;

    /// Returns the length of the parent.
    fn len(&self) -> usize;

    /// Returns whether the parent is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the child slots of the parent.
    fn slots(&self) -> &[Option<ArrayRef>];

    /// Returns `true` if the parent's encoding is `V`.
    ///
    /// Cheap encoding-id check that never constructs a view or materializes
    /// stack-backed parents.
    fn is_encoding<V: VTable>(&self) -> bool;

    /// Returns the `V`-typed encoding data if the parent's encoding is `V`.
    fn typed_data<V: VTable>(&self) -> Option<&V::TypedArrayData>;

    /// Returns a typed [`ParentView`] if the parent's encoding is `V`.
    ///
    /// The returned view borrows from `self`; stack-backed parents stay on the
    /// stack until a consumer explicitly calls
    /// [`ParentView::materialize_array_ref`].
    fn as_parent_view<V: VTable>(&self) -> Option<ParentView<'_, V>>;

    /// Does the parent match the given matcher.
    fn is<M: Matcher>(&self) -> bool
    where
        Self: Sized,
    {
        M::matches(self)
    }

    /// Returns the parent downcast by the given matcher, or `None` if it doesn't match.
    fn as_opt<M: Matcher>(&self) -> Option<M::Match<'_>>
    where
        Self: Sized,
    {
        M::try_match(self)
    }

    /// Returns the parent downcast by the given matcher, panicking if it doesn't match.
    fn as_<M: Matcher>(&self) -> M::Match<'_>
    where
        Self: Sized,
    {
        self.as_opt::<M>().vortex_expect("Failed to downcast")
    }
}

/// Trait for matching array types.
///
/// Matchers take any [`AsParent`] — a heap-allocated [`ArrayRef`] or a possibly
/// stack-allocated [`ParentRef`] — so one implementation serves both dispatch
/// chains. The returned match borrows from the parent and must not hide stack
/// materialization: matches expose an explicit
/// [`ParentView::materialize_array_ref`]-style hook instead of `AsRef<ArrayRef>`.
pub trait Matcher {
    /// The view type produced by a successful match, borrowing from the parent.
    type Match<'a>;

    /// Check if the given parent matches this matcher type.
    ///
    /// The default implementation delegates through [`try_match`](Self::try_match).
    /// Override when a cheaper check (e.g. an encoding-id comparison) suffices.
    fn matches<P: AsParent>(parent: &P) -> bool {
        Self::try_match(parent).is_some()
    }

    /// Try to match the parent, returning the matched view type if successful.
    ///
    /// The returned match borrows from `parent`, so matchers can return a
    /// [`ParentView`] without forcing the parent to materialize. Implementations
    /// typically delegate to [`AsParent::as_parent_view`] or [`AsParent::as_opt`].
    fn try_match<'a, P: AsParent>(parent: &'a P) -> Option<Self::Match<'a>>;
}

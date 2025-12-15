// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use crate::ArrayRef;
use crate::vtable::ArrayId;
use crate::vtable::VTable;

/// Trait for matching array types in optimizer rules
pub trait Matcher: Send + Sync + 'static {
    type View<'a>;

    /// Return the key for this matcher
    fn key(&self) -> MatchKey;

    /// Try to match the given array to this matcher type
    fn try_match<'a>(&self, array: &'a ArrayRef) -> Option<Self::View<'a>>;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MatchKey {
    Any,
    Array(ArrayId),
}

/// Matches any array type (wildcard matcher)
#[derive(Debug)]
pub struct AnyArray;

impl Matcher for AnyArray {
    type View<'a> = &'a ArrayRef;

    fn key(&self) -> MatchKey {
        MatchKey::Any
    }

    fn try_match<'a>(&self, array: &'a ArrayRef) -> Option<Self::View<'a>> {
        Some(array)
    }
}

/// Matches a specific Array by its encoding ID.
#[derive(Debug)]
pub struct Exact<V: VTable> {
    id: ArrayId,
    _phantom: PhantomData<V>,
}

impl<V: VTable> Matcher for Exact<V> {
    type View<'a> = &'a V::Array;

    fn key(&self) -> MatchKey {
        MatchKey::Array(self.id.clone())
    }

    fn try_match<'a>(&self, parent: &'a ArrayRef) -> Option<Self::View<'a>> {
        parent.as_opt::<V>()
    }
}

impl<V: VTable> Exact<V> {
    /// Create a new Exact matcher for the given ArrayId.
    ///
    /// # Safety
    ///
    /// The optimizer will attempt to downcast the array to the type V when matching.
    /// If an array with the given ID does not match type V, the rule will silently not be applied.
    pub unsafe fn new_unchecked(id: ArrayId) -> Self {
        Self {
            id,
            _phantom: PhantomData,
        }
    }
}

impl<V: VTable> From<&'static V> for Exact<V> {
    fn from(vtable: &'static V) -> Self {
        Self {
            id: vtable.id(),
            _phantom: PhantomData,
        }
    }
}

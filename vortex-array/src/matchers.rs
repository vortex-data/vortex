// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use crate::ArrayRef;
use crate::vtable::VTable;

/// Trait for matching array types in optimizer rules
pub trait Matcher: Send + Sync + 'static {
    type View<'a>;

    /// Try to match the given array to this matcher type
    fn try_match<'a>(&self, array: &'a ArrayRef) -> Option<Self::View<'a>>;
}

/// Matches any array type (wildcard matcher)
#[derive(Debug)]
pub struct AnyArray;

impl Matcher for AnyArray {
    type View<'a> = &'a ArrayRef;

    fn try_match<'a>(&self, array: &'a ArrayRef) -> Option<Self::View<'a>> {
        Some(array)
    }
}

/// Matches a specific Array by its VTable type.
#[derive(Debug)]
pub struct Exact<V: VTable>(PhantomData<V>);

impl<V: VTable> Exact<V> {
    /// Create a new Exact matcher for the given VTable type
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<V: VTable> Matcher for Exact<V> {
    type View<'a> = &'a V::Array;

    fn try_match<'a>(&self, parent: &'a ArrayRef) -> Option<Self::View<'a>> {
        parent.as_opt::<V>()
    }
}

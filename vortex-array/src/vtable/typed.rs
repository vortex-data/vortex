// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed array wrapper [`Array<V>`] that pairs a [`VTable`] instance with the inner array data.
//!
//! This is the new counterpart to [`ArrayAdapter<V>`](crate::ArrayAdapter), introduced as part of
//! the vtable migration. Unlike ArrayAdapter (which is `#[repr(transparent)]` over `V::Array`),
//! `Array<V>` stores the vtable instance alongside the inner data, enabling safe downcasting
//! through standard `Arc::downcast` instead of unsafe transmutes.
//!
//! During the migration, both `Array<V>` and `ArrayAdapter<V>` coexist behind `ArrayRef`. The
//! [`Matcher`](crate::matcher::Matcher) implementation tries both types when downcasting.

use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Deref;
use std::sync::Arc;

use crate::ArrayRef;
use crate::IntoArray;
use crate::vtable::VTable;

/// A typed array, parameterized by a concrete [`VTable`].
///
/// This struct holds the vtable instance alongside the encoding-specific data (`V::Array`).
/// It implements [`Deref`] to `V::Array`, so encoding-specific methods are callable directly
/// on `&Array<V>`.
///
/// Common array properties (dtype, len, stats) are still accessed through `V` methods on
/// the inner array during the transition period.
pub struct Array<V: VTable> {
    vtable: V,
    pub(crate) array: V::Array,
}

impl<V: VTable> Array<V> {
    /// Create a new typed array wrapping a vtable and inner array data.
    pub fn new(vtable: V, array: V::Array) -> Self {
        Self { vtable, array }
    }

    /// Returns a reference to the vtable.
    ///
    /// Note: this is intentionally named differently from `DynArray::vtable()` to avoid
    /// clippy::same_name_method. Use this for typed vtable access, or `DynArray::vtable()`
    /// for the dynamic vtable reference.
    pub fn typed_vtable(&self) -> &V {
        &self.vtable
    }

    /// Returns a reference to the inner encoding-specific array data.
    pub fn inner(&self) -> &V::Array {
        &self.array
    }

    /// Consumes this array and returns the inner encoding-specific data.
    pub fn into_inner(self) -> V::Array {
        self.array
    }
}

impl<V: VTable> Deref for Array<V> {
    type Target = V::Array;

    fn deref(&self) -> &V::Array {
        &self.array
    }
}

impl<V: VTable> Clone for Array<V> {
    fn clone(&self) -> Self {
        Self {
            vtable: self.vtable.clone(),
            array: self.array.clone(),
        }
    }
}

impl<V: VTable> Debug for Array<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Array")
            .field("encoding", &self.vtable.id())
            .field("inner", &self.array)
            .finish()
    }
}

impl<V: VTable> IntoArray for Array<V> {
    fn into_array(self) -> ArrayRef {
        Arc::new(self)
    }
}

impl<V: VTable> From<Array<V>> for ArrayRef {
    fn from(value: Array<V>) -> ArrayRef {
        value.into_array()
    }
}

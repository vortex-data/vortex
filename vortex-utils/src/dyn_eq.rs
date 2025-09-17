// SPDX-FileCopyrightText: 2016-2025 Copyright The Apache Software Foundation
// SPDX-FileCopyrightText: 2025 Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileComment: Derived from upstream file datafusion/expr-common/src/dyn_eq.rs at commit 3c021ac at https://github.com/apache/datafusion
// SPDX-FileNotice: https://github.com/apache/datafusion/blob/3c021aceb50546ed1a1235afd207df7cc70e6d00/datafusion/expr-common/src/dyn_eq.rs

//! A dyn-compatible version of `Eq` and `Hash` traits.

use std::any::Any;
use std::hash::{Hash, Hasher};

/// A dyn-compatible version of [`Eq`] trait.
/// The implementation constraints for this trait are the same as for [`Eq`]:
/// the implementation must be reflexive, symmetric, and transitive.
/// Additionally, if two values can be compared with [`DynEq`] and [`PartialEq`] then
/// they must be [`DynEq`]-equal if and only if they are [`PartialEq`]-equal.
/// It is therefore strongly discouraged to implement this trait for types
/// that implement `PartialEq<Other>` or `Eq<Other>` for any type `Other` other than `Self`.
///
/// Note: This trait should not be implemented directly. Implement `Eq` and `Any` and use
/// the blanket implementation.
#[allow(private_bounds)]
pub trait DynEq: private::EqSealed {
    /// Tests for self and other values to be equal.
    fn dyn_eq(&self, other: &dyn Any) -> bool;
}

impl<T: Eq + Any> private::EqSealed for T {}
impl<T: Eq + Any> DynEq for T {
    /// Tests for self and other values to be equal.
    fn dyn_eq(&self, other: &dyn Any) -> bool {
        other.downcast_ref::<Self>() == Some(self)
    }
}

/// A dyn-compatible version of [`Hash`] trait.
/// If two values are equal according to [`DynEq`], they must produce the same hash value.
///
/// Note: This trait should not be implemented directly. Implement `Hash` and `Any` and use
/// the blanket implementation.
#[allow(private_bounds)]
pub trait DynHash: private::HashSealed {
    /// Hash this value into the given state.
    fn dyn_hash(&self, _state: &mut dyn Hasher);
}

impl<T: Hash + Any> private::HashSealed for T {}
impl<T: Hash + Any> DynHash for T {
    fn dyn_hash(&self, mut state: &mut dyn Hasher) {
        self.type_id().hash(&mut state);
        self.hash(&mut state)
    }
}

mod private {
    pub(super) trait EqSealed {}
    pub(super) trait HashSealed {}
}

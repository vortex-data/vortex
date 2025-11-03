// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dynamic traits for equality and hashing.

use std::any::Any;

/// A dynamic equality trait for types implementing [`PartialEq`] or [`Eq`].
pub trait DynEq: Any + private::SealedEq {
    /// Compares `self` with another `Any` type for equality.
    fn dyn_eq(&self, other: &dyn Any) -> bool;
}

impl<T: PartialEq + 'static> DynEq for T {
    fn dyn_eq(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|other| other.eq(self))
    }
}

/// A dynamic hash trait for types implementing [`std::hash::Hash`].
pub trait DynHash: private::SealedHash {
    /// Hashes `self` into the given hasher.
    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher);
}

impl<T: std::hash::Hash + 'static> DynHash for T {
    fn dyn_hash(&self, mut state: &mut dyn std::hash::Hasher) {
        std::hash::Hash::hash(self, &mut state);
    }
}

mod private {
    pub trait SealedEq {}
    impl<T: PartialEq + ?Sized> SealedEq for T {}

    pub trait SealedHash {}
    impl<T: std::hash::Hash + ?Sized> SealedHash for T {}
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::fmt::Debug;

    use super::DynEq;

    trait MyTrait: DynEq + Debug + Any {}

    impl MyTrait for u32 {}

    impl MyTrait for u64 {}

    impl PartialEq for dyn MyTrait {
        fn eq(&self, other: &Self) -> bool {
            self.dyn_eq(other)
        }
    }

    #[test]
    fn test_dyn_eq() {
        let var_w = &2_u32 as &dyn MyTrait;
        let var_x = &5_u32 as &dyn MyTrait;
        let var_y = &5_u32 as &dyn MyTrait;
        let var_z = &5_u64 as &dyn MyTrait;

        // Same value, same type
        assert_eq!(var_x, var_y);
        // Different value, same type
        assert_ne!(var_w, var_x);
        // Same value, different type
        assert_ne!(var_y, var_z);
    }
}

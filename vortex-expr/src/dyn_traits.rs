// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

// TODO(adam): Look into having a similar setup as dyn-hash that will allow deriving `PartialEq` for structs
// that have `Arc<dyn Trait>` fields.
/// Allows comparing dyn-compatible objects, like [`ExprRef`](crate::ExprRef).
pub trait DynEq {
    fn dyn_eq(&self, other: &dyn Any) -> bool;
}

impl<T: Eq + Any> DynEq for T {
    fn dyn_eq(&self, other: &dyn Any) -> bool {
        Some(self) == other.downcast_ref::<Self>()
    }
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

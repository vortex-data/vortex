use std::ops::{BitAnd, Not};

use vortex_error::vortex_panic;

use crate::{AllOr, Mask};

impl BitAnd for &Mask {
    type Output = Mask;

    fn bitand(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            vortex_panic!("Masks must have the same length");
        }

        match (self.boolean_buffer(), rhs.boolean_buffer()) {
            (AllOr::All, _) => rhs.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) => Mask::new_false(self.len()),
            (_, AllOr::None) => Mask::new_false(self.len()),
            (AllOr::Some(lhs), AllOr::Some(rhs)) => Mask::from_buffer(lhs & rhs),
        }
    }
}

impl Not for &Mask {
    type Output = Mask;

    fn not(self) -> Self::Output {
        match self.boolean_buffer() {
            AllOr::All => Mask::new_false(self.len()),
            AllOr::None => Mask::new_true(self.len()),
            AllOr::Some(buffer) => Mask::from_buffer(!buffer),
        }
    }
}

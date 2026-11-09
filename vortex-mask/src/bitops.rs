// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::Not;

use vortex_error::vortex_panic;

use crate::AllOr;
use crate::Mask;

impl BitAnd for &Mask {
    type Output = Mask;

    fn bitand(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            vortex_panic!("Masks must have the same length");
        }

        match (self.bit_buffer(), rhs.bit_buffer()) {
            (AllOr::All, _) => rhs.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) => Mask::new_false(self.len()),
            (_, AllOr::None) => Mask::new_false(self.len()),
            (AllOr::Some(lhs), AllOr::Some(rhs)) => Mask::from_buffer(lhs & rhs),
        }
    }
}

impl BitAnd<&Mask> for Mask {
    type Output = Mask;

    /// Owned-left AND: can reuse the left buffer in-place when possible.
    fn bitand(self, rhs: &Mask) -> Self::Output {
        if self.len() != rhs.len() {
            vortex_panic!("Masks must have the same length");
        }

        match (self.bit_buffer(), rhs.bit_buffer()) {
            (AllOr::All, _) => rhs.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Mask::new_false(self.len()),
            (_, AllOr::All) => self,
            (AllOr::Some(_), AllOr::Some(rhs_buf)) => {
                Mask::from_buffer(self.into_bit_buffer() & rhs_buf)
            }
        }
    }
}

impl BitOr for &Mask {
    type Output = Mask;

    fn bitor(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            vortex_panic!("Masks must have the same length");
        }

        match (self.bit_buffer(), rhs.bit_buffer()) {
            (AllOr::All, _) => Mask::new_true(self.len()),
            (_, AllOr::All) => Mask::new_true(self.len()),
            (AllOr::None, _) => rhs.clone(),
            (_, AllOr::None) => self.clone(),
            (AllOr::Some(lhs), AllOr::Some(rhs)) => Mask::from_buffer(lhs | rhs),
        }
    }
}

impl BitOr<&Mask> for Mask {
    type Output = Mask;

    /// Owned-left OR: can reuse the left buffer in-place when possible.
    fn bitor(self, rhs: &Mask) -> Self::Output {
        if self.len() != rhs.len() {
            vortex_panic!("Masks must have the same length");
        }

        match (self.bit_buffer(), rhs.bit_buffer()) {
            (AllOr::All, _) | (_, AllOr::All) => Mask::new_true(self.len()),
            (AllOr::None, _) => rhs.clone(),
            (_, AllOr::None) => self,
            (AllOr::Some(_), AllOr::Some(rhs_buf)) => {
                Mask::from_buffer(self.into_bit_buffer() | rhs_buf)
            }
        }
    }
}

impl Mask {
    /// Computes `self & !rhs` (AND NOT), equivalent to set difference.
    pub fn bitand_not(self, rhs: &Mask) -> Mask {
        if self.len() != rhs.len() {
            vortex_panic!("Masks must have the same length");
        }
        match (self.bit_buffer(), rhs.bit_buffer()) {
            (AllOr::None, _) | (_, AllOr::All) => Mask::new_false(self.len()),
            (_, AllOr::None) => self,
            (AllOr::All, _) => !rhs,
            (AllOr::Some(_), AllOr::Some(rhs_buf)) => {
                Mask::from_buffer(self.into_bit_buffer().into_bitand_not(rhs_buf))
            }
        }
    }
}

impl Not for Mask {
    type Output = Mask;

    fn not(self) -> Self::Output {
        !(&self)
    }
}

impl Not for &Mask {
    type Output = Mask;

    fn not(self) -> Self::Output {
        match self.bit_buffer() {
            AllOr::All => Mask::new_false(self.len()),
            AllOr::None => Mask::new_true(self.len()),
            AllOr::Some(buffer) => Mask::from_buffer(!buffer),
        }
    }
}

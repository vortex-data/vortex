// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::{BitOr, Rem, Shl, Shr, Sub};

use arbitrary::unstructured::Int;
use num_traits::{CheckedAdd, WrappingAdd, WrappingSub};
use primitive_types::U256;

use crate::i256;

/// Used only internally to implement `Int` for i256
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct U256Wrapper(U256);

impl Sub for U256Wrapper {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        U256Wrapper(self.0 - rhs.0)
    }
}

impl Rem for U256Wrapper {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        U256Wrapper(self.0 % rhs.0)
    }
}

impl Shr for U256Wrapper {
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        U256Wrapper(self.0 >> rhs.0)
    }
}

impl Shl<usize> for U256Wrapper {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        U256Wrapper(self.0 << rhs)
    }
}

impl BitOr for U256Wrapper {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        U256Wrapper(self.0 | rhs.0)
    }
}

impl Int for U256Wrapper {
    type Unsigned = Self;

    const ZERO: Self = U256Wrapper(U256::zero());
    const ONE: Self = U256Wrapper(U256::one());
    const MAX: Self = U256Wrapper(U256::max_value());

    fn from_u8(b: u8) -> Self {
        Self(U256::from(b))
    }

    fn from_usize(u: usize) -> Self {
        Self(U256::from(u))
    }

    fn checked_add(self, rhs: Self) -> Option<Self> {
        self.0.checked_add(rhs.0).map(U256Wrapper)
    }

    fn wrapping_add(self, rhs: Self) -> Self {
        let (v, _) = self.0.overflowing_add(rhs.0);
        U256Wrapper(v)
    }

    fn wrapping_sub(self, rhs: Self) -> Self {
        let (v, _) = self.0.overflowing_sub(rhs.0);
        U256Wrapper(v)
    }

    fn to_unsigned(self) -> Self::Unsigned {
        self
    }

    fn from_unsigned(unsigned: Self::Unsigned) -> Self {
        unsigned
    }
}

#[allow(clippy::same_name_method)]
impl Int for i256 {
    type Unsigned = U256Wrapper;
    const ZERO: Self = i256::ZERO;
    const ONE: Self = i256::ONE;
    const MAX: Self = i256::MAX;

    fn from_u8(b: u8) -> Self {
        Self::from_i128(b as i128)
    }

    fn from_usize(u: usize) -> Self {
        Self::from_i128(u as i128)
    }

    fn checked_add(self, rhs: Self) -> Option<Self> {
        <Self as CheckedAdd>::checked_add(&self, &rhs)
    }

    fn wrapping_add(self, rhs: Self) -> Self {
        <Self as WrappingAdd>::wrapping_add(&self, &rhs)
    }

    fn wrapping_sub(self, rhs: Self) -> Self {
        <Self as WrappingSub>::wrapping_sub(&self, &rhs)
    }

    fn to_unsigned(self) -> Self::Unsigned {
        let bytes = self.to_le_bytes(); // or to_be_bytes(), depends on your impl
        U256Wrapper(U256::from_little_endian(&bytes))
    }

    fn from_unsigned(unsigned: Self::Unsigned) -> Self {
        let bytes = unsigned.0.to_little_endian();
        i256::from_le_bytes(bytes)
    }
}

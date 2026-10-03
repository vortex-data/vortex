// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Order-preserving maps from numbers to unsigned "latent" integers.

use std::fmt::Debug;
use std::ops::BitAnd;

/// An unsigned integer that the codec bins and entropy codes.
pub trait Latent:
    Copy + Ord + Default + Debug + Send + Sync + BitAnd<Output = Self> + 'static
{
    const BITS: u32;
    /// The midpoint of the latent range, used to center deltas so small negative and positive
    /// differences land next to each other.
    const MID: Self;
    const ZERO: Self;

    fn to_u64(self) -> u64;
    /// Truncating conversion.
    fn from_u64(v: u64) -> Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
}

macro_rules! impl_latent {
    ($t:ty) => {
        impl Latent for $t {
            const BITS: u32 = <$t>::BITS;
            const MID: Self = 1 << (<$t>::BITS - 1);
            const ZERO: Self = 0;

            #[inline]
            fn to_u64(self) -> u64 {
                u64::from(self)
            }

            #[inline]
            #[allow(clippy::cast_possible_truncation)]
            fn from_u64(v: u64) -> Self {
                v as $t
            }

            #[inline]
            fn wrapping_add(self, other: Self) -> Self {
                <$t>::wrapping_add(self, other)
            }

            #[inline]
            fn wrapping_sub(self, other: Self) -> Self {
                <$t>::wrapping_sub(self, other)
            }
        }
    };
}

impl_latent!(u8);
impl_latent!(u16);
impl_latent!(u32);
impl_latent!(u64);

/// A number type the codec can compress, with an order-preserving bijection to its latent.
pub trait Number: Copy + Send + Sync + 'static {
    type L: Latent;

    fn to_latent(self) -> Self::L;
    fn from_latent(latent: Self::L) -> Self;
}

macro_rules! impl_unsigned {
    ($t:ty) => {
        impl Number for $t {
            type L = $t;

            #[inline]
            fn to_latent(self) -> $t {
                self
            }

            #[inline]
            fn from_latent(latent: $t) -> Self {
                latent
            }
        }
    };
}

impl_unsigned!(u8);
impl_unsigned!(u16);
impl_unsigned!(u32);
impl_unsigned!(u64);

macro_rules! impl_signed {
    ($t:ty, $u:ty) => {
        impl Number for $t {
            type L = $u;

            #[inline]
            #[allow(clippy::cast_sign_loss)]
            fn to_latent(self) -> $u {
                (self as $u) ^ <$u as Latent>::MID
            }

            #[inline]
            #[allow(clippy::cast_possible_wrap)]
            fn from_latent(latent: $u) -> Self {
                (latent ^ <$u as Latent>::MID) as $t
            }
        }
    };
}

impl_signed!(i8, u8);
impl_signed!(i16, u16);
impl_signed!(i32, u32);
impl_signed!(i64, u64);

// Branch-free: negative floats flip every bit, others flip only the sign bit, so the loops that
// map whole blocks vectorize.
macro_rules! impl_float {
    ($t:ty, $u:ty, $i:ty) => {
        impl Number for $t {
            type L = $u;

            #[inline]
            #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
            fn to_latent(self) -> $u {
                let bits = self.to_bits();
                let sign_fill = ((bits as $i) >> (<$u>::BITS - 1)) as $u;
                bits ^ (sign_fill | <$u as Latent>::MID)
            }

            #[inline]
            #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
            fn from_latent(latent: $u) -> Self {
                let sign_fill = ((!latent as $i) >> (<$u>::BITS - 1)) as $u;
                <$t>::from_bits(latent ^ (sign_fill | <$u as Latent>::MID))
            }
        }
    };
}

impl_float!(f32, u32, i32);
impl_float!(f64, u64, i64);

impl Number for half::f16 {
    type L = u16;

    #[inline]
    fn to_latent(self) -> u16 {
        let bits = self.to_bits();
        if bits & u16::MID != 0 {
            !bits
        } else {
            bits | u16::MID
        }
    }

    #[inline]
    fn from_latent(latent: u16) -> Self {
        let bits = if latent & u16::MID != 0 {
            latent & !u16::MID
        } else {
            !latent
        };
        half::f16::from_bits(bits)
    }
}

/// Bits needed to represent any offset in `0..=range`.
#[inline]
pub(crate) fn offset_bits(range: u64) -> u8 {
    #[allow(clippy::cast_possible_truncation)]
    let bits = (u64::BITS - range.leading_zeros()) as u8;
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_order_preserved() {
        let xs = [i32::MIN, -5, -1, 0, 1, 7, i32::MAX];
        let ls: Vec<u32> = xs.iter().map(|x| x.to_latent()).collect();
        assert!(ls.windows(2).all(|w| w[0] < w[1]));
        for x in xs {
            assert_eq!(i32::from_latent(x.to_latent()), x);
        }
    }

    #[test]
    fn float_order_preserved() {
        let xs = [f64::NEG_INFINITY, -2.5, -0.0, 0.0, 1e-300, 3.25, f64::INFINITY];
        let ls: Vec<u64> = xs.iter().map(|x| x.to_latent()).collect();
        assert!(ls.windows(2).all(|w| w[0] < w[1]));
        for x in xs {
            assert_eq!(f64::from_latent(x.to_latent()).to_bits(), x.to_bits());
        }
        let nan = f64::from_bits(0x7ff8_0000_dead_beef);
        assert_eq!(f64::from_latent(nan.to_latent()).to_bits(), nan.to_bits());
    }
}

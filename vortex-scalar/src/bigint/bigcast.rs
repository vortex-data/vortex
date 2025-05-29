use crate::i256;

/// Checked conversion from one primitive type to another.
///
/// This is meant to mirror the `ToPrimitive` trait from `num-traits` but with awareness of `i256`.
pub trait ToPrimitive: num_traits::ToPrimitive {
    /// Converts the value of `self` to an `i256`. If the value cannot be
    /// represented by an `i256`, then `None` is returned.
    fn to_i256(&self) -> Option<i256>;
}

// Implementation for primitive types that already implement ToPrimitive from num-traits.
macro_rules! impl_toprimitive_lossless {
    ($typ:ty) => {
        impl ToPrimitive for $typ {
            #[inline]
            fn to_i256(&self) -> Option<i256> {
                Some(i256::from_i128(*self as i128))
            }
        }
    };
}

// unsigned, except for u128, all losslessly cast into i128
impl_toprimitive_lossless!(u8);
impl_toprimitive_lossless!(u16);
impl_toprimitive_lossless!(u32);
impl_toprimitive_lossless!(u64);

// signed all losslessly cast into i128
impl_toprimitive_lossless!(i8);
impl_toprimitive_lossless!(i16);
impl_toprimitive_lossless!(i32);
impl_toprimitive_lossless!(i64);
impl_toprimitive_lossless!(i128);

// u128 -> i256 always lossless
impl ToPrimitive for u128 {
    fn to_i256(&self) -> Option<i256> {
        Some(i256::from_parts(*self, 0))
    }
}

// identity
impl ToPrimitive for i256 {
    fn to_i256(&self) -> Option<i256> {
        Some(*self)
    }
}

/// Checked numeric casts up to and including i256 support.
///
/// This is meant as a more inclusive version of `NumCast` from the `num-traits` crate.
pub trait BigCast: Sized + ToPrimitive {
    /// Cast the value `n` to Self using the relevant `ToPrimitive` method. If the value cannot
    /// be represented by Self, `None` is returned.
    fn from<T: ToPrimitive>(n: T) -> Option<Self>;
}

macro_rules! impl_big_cast {
    ($T:ty, $conv:ident) => {
        impl BigCast for $T {
            fn from<T: ToPrimitive>(n: T) -> Option<Self> {
                n.$conv()
            }
        }
    };
}

impl_big_cast!(u8, to_u8);
impl_big_cast!(u16, to_u16);
impl_big_cast!(u32, to_u32);
impl_big_cast!(u64, to_u64);
impl_big_cast!(u128, to_u128);
impl_big_cast!(i8, to_i8);
impl_big_cast!(i16, to_i16);
impl_big_cast!(i32, to_i32);
impl_big_cast!(i64, to_i64);
impl_big_cast!(i128, to_i128);
impl_big_cast!(i256, to_i256);

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use rstest::rstest;

    use crate::{BigCast, i256};

    // All BigCast types must losslessly round-trip themselves
    #[rstest]
    #[case(u8::MAX)]
    #[case(u16::MAX)]
    #[case(u32::MAX)]
    #[case(u64::MAX)]
    #[case(u128::MAX)]
    #[case(i8::MAX)]
    #[case(i16::MAX)]
    #[case(i32::MAX)]
    #[case(i64::MAX)]
    #[case(i128::MAX)]
    #[case(i256::MAX)]
    fn test_big_cast_identity<T: BigCast + Eq + Debug + Copy>(#[case] n: T) {
        assert_eq!(<T as BigCast>::from(n).unwrap(), n);
    }

    macro_rules! test_big_cast_overflow {
        ($name:ident, $src:ty => $dst:ty, $max:expr, $one:expr) => {
            #[test]
            fn $name() {
                // lossless upcast of max
                let v = <$dst as BigCast>::from($max).unwrap();
                // Downcast must be lossless.
                assert_eq!(<$src as BigCast>::from(v), Some($max));

                // add one -> out of the bounds of the lower type
                let v = v + $one;
                assert_eq!(<$src as BigCast>::from(v), None);
            }
        };
    }

    test_big_cast_overflow!(test_i8_overflow, i8 => i16, i8::MAX, 1i16);
    test_big_cast_overflow!(test_i16_overflow, i16 => i32, i16::MAX, 1i32);
    test_big_cast_overflow!(test_i32_overflow, i32 => i64, i32::MAX, 1i64);
    test_big_cast_overflow!(test_i64_overflow, i64 => i128, i64::MAX, 1i128);
    test_big_cast_overflow!(test_i128_overflow, i128 => i256, i128::MAX, i256::ONE);
}

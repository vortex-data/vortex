// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bigcast;

use std::fmt::Display;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::BitOr;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Rem;
use std::ops::Shl;
use std::ops::Shr;
use std::ops::Sub;

pub use bigcast::*;
use num_traits::CheckedAdd;
use num_traits::CheckedDiv;
use num_traits::CheckedMul;
use num_traits::CheckedSub;
use num_traits::ConstZero;
use num_traits::One;
use num_traits::WrappingAdd;
use num_traits::WrappingSub;
use num_traits::Zero;
use vortex_error::VortexExpect;

/// Signed 256-bit integer type.
///
/// This is one of the physical representations of `DecimalScalar` values and can be safely converted
/// back and forth with Arrow's [`i256`][arrow_buffer::i256].
#[repr(transparent)]
#[expect(
    non_camel_case_types,
    reason = "i256 matches Rust primitive naming convention"
)]
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct i256(arrow_buffer::i256);

#[cfg(feature = "cudarc")]
unsafe impl cudarc::driver::DeviceRepr for i256 {}

#[expect(
    clippy::same_name_method,
    reason = "inherent methods intentionally shadow arrow_buffer::i256 methods"
)]
impl i256 {
    /// The zero value for `i256`.
    pub const ZERO: Self = Self(arrow_buffer::i256::ZERO);
    /// The one value for `i256`.
    pub const ONE: Self = Self(arrow_buffer::i256::ONE);
    /// The maximum value for `i256`.
    pub const MAX: Self = Self(arrow_buffer::i256::MAX);
    /// The minimum value for `i256`.
    pub const MIN: Self = Self(arrow_buffer::i256::MIN);

    /// Construct a new `i256` from an unsigned `lower` bits and a signed `upper` bits.
    pub const fn from_parts(lower: u128, upper: i128) -> Self {
        Self(arrow_buffer::i256::from_parts(lower, upper))
    }

    /// Create an `i256` value from a signed 128-bit value.
    pub const fn from_i128(i: i128) -> Self {
        Self(arrow_buffer::i256::from_i128(i))
    }

    /// Attempts to convert this i256 to an i128.
    ///
    /// Returns None if the value is too large to fit in an i128.
    pub fn maybe_i128(self) -> Option<i128> {
        self.0.to_i128()
    }

    /// Create an integer value from its little-endian byte array representation.
    pub const fn from_le_bytes(bytes: [u8; 32]) -> Self {
        Self(arrow_buffer::i256::from_le_bytes(bytes))
    }

    /// Split the 256-bit signed integer value into an unsigned lower bits and a signed upper bits.
    ///
    /// This version gives us ownership of the value.
    pub const fn into_parts(self) -> (u128, i128) {
        self.0.to_parts()
    }

    /// Split the 256-bit signed integer value into an unsigned lower bits and a signed upper bits.
    pub const fn to_parts(&self) -> (u128, i128) {
        self.0.to_parts()
    }

    /// Raises self to the power of `exp`, wrapping around on overflow.
    pub fn wrapping_pow(&self, exp: u32) -> Self {
        Self(self.0.wrapping_pow(exp))
    }

    /// Raises self to the power of `exp`, wrapping around on overflow.
    pub fn checked_pow(&self, exp: u32) -> Option<Self> {
        self.0.checked_pow(exp).map(Self)
    }

    /// Wrapping (modular) addition. Computes `self + other`, wrapping around at the boundary.
    pub fn wrapping_add(&self, other: Self) -> Self {
        Self(self.0.wrapping_add(other.0))
    }

    /// Return the memory representation of this integer as a byte array in little-endian byte order.
    #[inline]
    pub const fn to_le_bytes(&self) -> [u8; 32] {
        self.0.to_le_bytes()
    }

    /// Return the memory representation of this integer as a byte array in big-endian byte order.
    #[inline]
    pub const fn to_be_bytes(&self) -> [u8; 32] {
        self.0.to_be_bytes()
    }
}

impl From<i256> for arrow_buffer::i256 {
    fn from(i: i256) -> Self {
        i.0
    }
}

impl From<arrow_buffer::i256> for i256 {
    fn from(i: arrow_buffer::i256) -> Self {
        Self(i)
    }
}

impl Display for i256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add for i256 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.add(rhs.0))
    }
}

impl Sub for i256 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.sub(rhs.0))
    }
}

impl Neg for i256 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0.neg())
    }
}

impl Mul<Self> for i256 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0))
    }
}

impl Div<Self> for i256 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0.div(rhs.0))
    }
}

impl Rem<Self> for i256 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0.rem(rhs.0))
    }
}

impl Zero for i256 {
    fn zero() -> Self {
        Self::default()
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl ConstZero for i256 {
    const ZERO: Self = Self(arrow_buffer::i256::ZERO);
}

impl One for i256 {
    fn one() -> Self {
        Self(arrow_buffer::i256::ONE)
    }
}

impl CheckedAdd for i256 {
    fn checked_add(&self, v: &Self) -> Option<Self> {
        self.0.checked_add(v.0).map(Self)
    }
}

impl WrappingAdd for i256 {
    fn wrapping_add(&self, v: &Self) -> Self {
        Self(self.0.wrapping_add(v.0))
    }
}

impl CheckedSub for i256 {
    fn checked_sub(&self, v: &Self) -> Option<Self> {
        self.0.checked_sub(v.0).map(Self)
    }
}

impl WrappingSub for i256 {
    fn wrapping_sub(&self, v: &Self) -> Self {
        Self(self.0.wrapping_sub(v.0))
    }
}

impl CheckedMul for i256 {
    fn checked_mul(&self, v: &Self) -> Option<Self> {
        self.0.checked_mul(v.0).map(Self)
    }
}

impl CheckedDiv for i256 {
    fn checked_div(&self, v: &Self) -> Option<Self> {
        self.0.checked_div(v.0).map(Self)
    }
}

impl Shr<Self> for i256 {
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        use num_traits::ToPrimitive;

        Self(
            self.0.shr(
                rhs.0
                    .to_u8()
                    .vortex_expect("Can't shift more than 256 bits"),
            ),
        )
    }
}

impl Shl<usize> for i256 {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        use num_traits::ToPrimitive;
        Self(
            self.0
                .shl(rhs.to_u8().vortex_expect("Can't shift more than 256 bits")),
        )
    }
}

impl BitOr<Self> for i256 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0.bitor(rhs.0))
    }
}

impl AddAssign for i256 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl num_traits::ToPrimitive for i256 {
    fn to_i64(&self) -> Option<i64> {
        self.maybe_i128().and_then(|v| v.to_i64())
    }

    fn to_i128(&self) -> Option<i128> {
        self.maybe_i128()
    }

    fn to_u64(&self) -> Option<u64> {
        self.maybe_i128().and_then(|v| v.to_u64())
    }

    fn to_u128(&self) -> Option<u128> {
        self.maybe_i128().and_then(|v| v.to_u128())
    }
}

macro_rules! define_as_primitive {
    ($native_ty:ty) => {
        impl num_traits::AsPrimitive<i256> for $native_ty {
            fn as_(self) -> i256 {
                i256::from_i128(self as i128)
            }
        }

        impl num_traits::AsPrimitive<$native_ty> for i256 {
            #[allow(clippy::cast_possible_truncation)]
            fn as_(self) -> $native_ty {
                self.0.as_i128() as $native_ty
            }
        }
    };
}

impl num_traits::AsPrimitive<i256> for i256 {
    fn as_(self) -> i256 {
        self
    }
}

define_as_primitive!(i8);
define_as_primitive!(i16);
define_as_primitive!(i32);
define_as_primitive!(i64);
define_as_primitive!(i128);

#[cfg(test)]
#[expect(clippy::many_single_char_names)]
mod tests {
    use num_traits::ToPrimitive;

    use super::*;

    #[test]
    fn test_i256_constants() {
        assert_eq!(i256::ZERO, i256::from_i128(0));
        assert_eq!(i256::ONE, i256::from_i128(1));
        assert!(i256::MIN < i256::ZERO);
        assert!(i256::MAX > i256::ZERO);
        assert!(i256::MIN < i256::MAX);
    }

    #[test]
    fn test_i256_from_i128() {
        let value = i256::from_i128(123456789);
        assert_eq!(value.maybe_i128(), Some(123456789));

        let negative = i256::from_i128(-987654321);
        assert_eq!(negative.maybe_i128(), Some(-987654321));

        let max_i128 = i256::from_i128(i128::MAX);
        assert_eq!(max_i128.maybe_i128(), Some(i128::MAX));

        let min_i128 = i256::from_i128(i128::MIN);
        assert_eq!(min_i128.maybe_i128(), Some(i128::MIN));
    }

    #[test]
    fn test_i256_from_parts() {
        let value = i256::from_parts(1000, 2000);
        let (lower, upper) = value.into_parts();
        assert_eq!(lower, 1000);
        assert_eq!(upper, 2000);

        // Test to_parts (non-consuming)
        let (lower2, upper2) = value.to_parts();
        assert_eq!(lower2, 1000);
        assert_eq!(upper2, 2000);
    }

    #[test]
    fn test_i256_byte_conversions() {
        let original = i256::from_i128(123456789012345);

        // Test little-endian
        let le_bytes = original.to_le_bytes();
        let recovered_le = i256::from_le_bytes(le_bytes);
        assert_eq!(original, recovered_le);

        // Test big-endian
        let be_bytes = original.to_be_bytes();
        assert_ne!(le_bytes, be_bytes); // Should be different unless value is symmetric

        // Test zero
        let zero_le = i256::ZERO.to_le_bytes();
        assert_eq!(zero_le, [0u8; 32]);
    }

    #[test]
    fn test_i256_display() {
        let value = i256::from_i128(42);
        assert_eq!(format!("{value}"), "42");

        let negative = i256::from_i128(-42);
        assert_eq!(format!("{negative}"), "-42");
    }

    #[test]
    fn test_i256_arithmetic_add() {
        let a = i256::from_i128(100);
        let b = i256::from_i128(200);
        let sum = a + b;
        assert_eq!(sum.maybe_i128(), Some(300));

        // Test negative addition
        let c = i256::from_i128(-50);
        let sum2 = a + c;
        assert_eq!(sum2.maybe_i128(), Some(50));
    }

    #[test]
    fn test_i256_arithmetic_sub() {
        let a = i256::from_i128(500);
        let b = i256::from_i128(200);
        let diff = a - b;
        assert_eq!(diff.maybe_i128(), Some(300));

        // Test negative result
        let diff2 = b - a;
        assert_eq!(diff2.maybe_i128(), Some(-300));
    }

    #[test]
    fn test_i256_arithmetic_mul() {
        let a = i256::from_i128(100);
        let b = i256::from_i128(200);
        let product = a * b;
        assert_eq!(product.maybe_i128(), Some(20000));

        // Test negative multiplication
        let c = i256::from_i128(-5);
        let product2 = a * c;
        assert_eq!(product2.maybe_i128(), Some(-500));
    }

    #[test]
    fn test_i256_arithmetic_div() {
        let a = i256::from_i128(1000);
        let b = i256::from_i128(25);
        let quotient = a / b;
        assert_eq!(quotient.maybe_i128(), Some(40));

        // Test negative division
        let c = i256::from_i128(-1000);
        let quotient2 = c / b;
        assert_eq!(quotient2.maybe_i128(), Some(-40));
    }

    #[test]
    fn test_i256_arithmetic_rem() {
        let a = i256::from_i128(103);
        let b = i256::from_i128(10);
        let remainder = a % b;
        assert_eq!(remainder.maybe_i128(), Some(3));

        // Test negative remainder
        let c = i256::from_i128(-103);
        let remainder2 = c % b;
        assert_eq!(remainder2.maybe_i128(), Some(-3));
    }

    #[test]
    fn test_i256_wrapping_pow() {
        let base = i256::from_i128(2);
        let result = base.wrapping_pow(10);
        assert_eq!(result.maybe_i128(), Some(1024));

        let base2 = i256::from_i128(10);
        let result2 = base2.wrapping_pow(3);
        assert_eq!(result2.maybe_i128(), Some(1000));

        // Test with 0 exponent
        let result3 = base.wrapping_pow(0);
        assert_eq!(result3.maybe_i128(), Some(1));
    }

    #[test]
    fn test_i256_wrapping_add() {
        let a = i256::from_i128(100);
        let b = i256::from_i128(200);
        let result = a.wrapping_add(b);
        assert_eq!(result.maybe_i128(), Some(300));

        // Test the method version
        let result2 = a.wrapping_add(b);
        assert_eq!(result2.maybe_i128(), Some(300));
    }

    #[test]
    fn test_i256_zero_trait() {
        assert!(i256::zero().is_zero());
        assert!(!i256::from_i128(1).is_zero());
        assert!(!i256::from_i128(-1).is_zero());

        // Test ConstZero
        assert_eq!(i256::ZERO, <i256 as ConstZero>::ZERO);
    }

    #[test]
    fn test_i256_one_trait() {
        assert_eq!(i256::one(), i256::from_i128(1));
        assert!(!i256::one().is_zero());
    }

    #[test]
    fn test_i256_checked_add() {
        let a = i256::from_i128(100);
        let b = i256::from_i128(200);
        let result = a.checked_add(&b);
        assert_eq!(result, Some(i256::from_i128(300)));

        // Note: Testing overflow would require values larger than i128
    }

    #[test]
    fn test_i256_wrapping_add_trait() {
        let a = i256::from_i128(100);
        let b = i256::from_i128(200);
        let result = <i256 as WrappingAdd>::wrapping_add(&a, &b);
        assert_eq!(result.maybe_i128(), Some(300));
    }

    #[test]
    fn test_i256_checked_sub() {
        let a = i256::from_i128(500);
        let b = i256::from_i128(200);
        let result = a.checked_sub(&b);
        assert_eq!(result, Some(i256::from_i128(300)));

        // Test negative result
        let result2 = b.checked_sub(&a);
        assert_eq!(result2, Some(i256::from_i128(-300)));
    }

    #[test]
    fn test_i256_wrapping_sub_trait() {
        let a = i256::from_i128(500);
        let b = i256::from_i128(200);
        let result = <i256 as WrappingSub>::wrapping_sub(&a, &b);
        assert_eq!(result.maybe_i128(), Some(300));
    }

    #[test]
    fn test_i256_shift_right() {
        let value = i256::from_i128(128);
        let shift_amount = i256::from_i128(1);
        let result = value >> shift_amount;
        assert_eq!(result.maybe_i128(), Some(64));

        let shift_amount2 = i256::from_i128(2);
        let result2 = value >> shift_amount2;
        assert_eq!(result2.maybe_i128(), Some(32));

        // Shift by 0
        let shift_zero = i256::from_i128(0);
        let result3 = value >> shift_zero;
        assert_eq!(result3.maybe_i128(), Some(128));
    }

    #[test]
    fn test_i256_shift_left() {
        let value = i256::from_i128(32);
        let result = value << 1;
        assert_eq!(result.maybe_i128(), Some(64));

        let result2 = value << 2;
        assert_eq!(result2.maybe_i128(), Some(128));

        // Shift by 0
        let result3 = value << 0;
        assert_eq!(result3.maybe_i128(), Some(32));
    }

    #[test]
    fn test_i256_bitor() {
        let a = i256::from_i128(0b1010);
        let b = i256::from_i128(0b1100);
        let result = a | b;
        assert_eq!(result.maybe_i128(), Some(0b1110));

        // Test with zero
        let result2 = a | i256::ZERO;
        assert_eq!(result2.maybe_i128(), Some(0b1010));
    }

    #[test]
    fn test_i256_to_primitive() {
        let value = i256::from_i128(1000);

        // Test to_i64
        assert_eq!(value.to_i64(), Some(1000i64));

        // Test to_i128
        assert_eq!(value.to_i128(), Some(1000i128));

        // Test to_u64
        assert_eq!(value.to_u64(), Some(1000u64));

        // Test to_u128
        assert_eq!(value.to_u128(), Some(1000u128));

        // Test negative value
        let negative = i256::from_i128(-500);
        assert_eq!(negative.to_i64(), Some(-500i64));
        assert_eq!(negative.to_i128(), Some(-500i128));
        assert_eq!(negative.to_u64(), None); // Can't convert negative to unsigned
        assert_eq!(negative.to_u128(), None);
    }

    #[test]
    fn test_i256_arrow_buffer_conversion() {
        let arrow_value = arrow_buffer::i256::from_i128(42);
        let our_value: i256 = arrow_value.into();
        assert_eq!(our_value.maybe_i128(), Some(42));

        // Convert back
        let arrow_again: arrow_buffer::i256 = our_value.into();
        assert_eq!(arrow_again, arrow_value);
    }

    #[test]
    fn test_i256_default() {
        let default_value = i256::default();
        assert_eq!(default_value, i256::ZERO);
        assert!(default_value.is_zero());
    }

    #[test]
    fn test_i256_ordering() {
        let a = i256::from_i128(100);
        let b = i256::from_i128(200);
        let c = i256::from_i128(-50);

        assert!(a < b);
        assert!(b > a);
        assert!(c < a);
        assert!(c < b);
        assert_eq!(a, a);
        assert_ne!(a, b);
    }

    #[test]
    fn test_i256_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash;
        use std::hash::Hasher;

        let value1 = i256::from_i128(42);
        let value2 = i256::from_i128(42);
        let value3 = i256::from_i128(43);

        let mut hasher1 = DefaultHasher::new();
        value1.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        value2.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        let mut hasher3 = DefaultHasher::new();
        value3.hash(&mut hasher3);
        let hash3 = hasher3.finish();

        assert_eq!(hash1, hash2); // Same values should have same hash
        assert_ne!(hash1, hash3); // Different values should (likely) have different hash
    }

    #[test]
    fn test_i256_large_value_loses_precision() {
        // Create a value that doesn't fit in i128
        let large_value = i256::from_parts(u128::MAX, 1);
        assert_eq!(large_value.maybe_i128(), None);

        // The parts should be preserved
        let (lower, upper) = large_value.to_parts();
        assert_eq!(lower, u128::MAX);
        assert_eq!(upper, 1);
    }
}

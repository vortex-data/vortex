// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`DecimalValue`] type representing a typed decimal value.

use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;

use num_traits::CheckedAdd;
use num_traits::CheckedDiv;
use num_traits::CheckedMul;
use num_traits::CheckedSub;
use vortex_error::VortexExpect;

use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::ToI256;
use crate::dtype::i256;
use crate::match_each_decimal_value;
use crate::match_each_decimal_value_type;

/// Widens both operands to the larger of their two decimal types, then applies the checked op.
macro_rules! checked_widening_binary_op {
    ($self:expr, $other:expr, $op:path) => {{
        let target = $self.decimal_type().max($other.decimal_type());
        match_each_decimal_value_type!(target, |T| {
            let a: T = $self
                .cast()
                .vortex_expect("widening cast to wider decimal type must always succeed");
            let b: T = $other
                .cast()
                .vortex_expect("widening cast to wider decimal type must always succeed");
            Some(DecimalValue::from($op(&a, &b)?))
        })
    }};
}

/// A decimal value that can be stored in various integer widths.
///
/// This enum represents decimal values with different storage sizes,
/// from 8-bit to 256-bit integers.
#[derive(Debug, Clone, Copy)]
pub enum DecimalValue {
    /// 8-bit signed decimal value.
    I8(i8),
    /// 16-bit signed decimal value.
    I16(i16),
    /// 32-bit signed decimal value.
    I32(i32),
    /// 64-bit signed decimal value.
    I64(i64),
    /// 128-bit signed decimal value.
    I128(i128),
    /// 256-bit signed decimal value.
    I256(i256),
}

impl DecimalValue {
    /// Cast `self` to T using the respective `ToPrimitive` method.
    /// If the value cannot be represented by `T`, `None` is returned.
    pub fn cast<T: NativeDecimalType>(&self) -> Option<T> {
        match_each_decimal_value!(self, |value| { T::from(*value) })
    }

    /// Returns a reasonable precision and scale as a [`DecimalDType`] for the given
    /// [`DecimalValue`].
    ///
    /// Note that this is **not** the same as [`DecimalValue::decimal_type`]!!!
    pub fn decimal_dtype(&self) -> DecimalDType {
        // Default to a reasonable precision and scale.
        match self {
            DecimalValue::I8(_) => DecimalDType::new(3, 0),
            DecimalValue::I16(_) => DecimalDType::new(5, 0),
            DecimalValue::I32(_) => DecimalDType::new(10, 0),
            DecimalValue::I64(_) => DecimalDType::new(19, 0),
            DecimalValue::I128(_) => DecimalDType::new(38, 0),
            DecimalValue::I256(_) => DecimalDType::new(76, 0),
        }
    }

    /// Returns the [`DecimalType`] for the given [`DecimalValue`].
    ///
    /// Note that this is **not** the same as [`DecimalValue::decimal_dtype`]!!!
    pub fn decimal_type(&self) -> DecimalType {
        match self {
            DecimalValue::I8(_) => DecimalType::I8,
            DecimalValue::I16(_) => DecimalType::I16,
            DecimalValue::I32(_) => DecimalType::I32,
            DecimalValue::I64(_) => DecimalType::I64,
            DecimalValue::I128(_) => DecimalType::I128,
            DecimalValue::I256(_) => DecimalType::I256,
        }
    }

    /// Returns true if this decimal value is zero.
    pub fn is_zero(&self) -> bool {
        match self {
            DecimalValue::I8(v) => *v == 0,
            DecimalValue::I16(v) => *v == 0,
            DecimalValue::I32(v) => *v == 0,
            DecimalValue::I64(v) => *v == 0,
            DecimalValue::I128(v) => *v == 0,
            DecimalValue::I256(v) => *v == i256::ZERO,
        }
    }

    /// Convert this `DecimalValue` to an [`i256`].
    ///
    /// This conversion always succeeds since [`i256`] can represent every stored variant.
    pub fn as_i256(&self) -> i256 {
        match_each_decimal_value!(self, |v| {
            v.to_i256()
                .vortex_expect("upcast to i256 must always succeed")
        })
    }

    /// Returns the 0 value given the [`DecimalType`].
    pub fn zero(decimal_type: &DecimalDType) -> Self {
        let smallest_type = DecimalType::smallest_decimal_value_type(decimal_type);

        match smallest_type {
            DecimalType::I8 => DecimalValue::I8(0),
            DecimalType::I16 => DecimalValue::I16(0),
            DecimalType::I32 => DecimalValue::I32(0),
            DecimalType::I64 => DecimalValue::I64(0),
            DecimalType::I128 => DecimalValue::I128(0),
            DecimalType::I256 => DecimalValue::I256(i256::ZERO),
        }
    }

    /// Check if this decimal value fits within the precision constraints of the given decimal type.
    ///
    /// The precision defines the total number of significant digits that can be represented.
    /// The stored value (regardless of scale) must fit within the range defined by precision.
    /// For precision P, the maximum absolute stored value is 10^P - 1.
    pub fn fits_in_precision(&self, decimal_type: DecimalDType) -> bool {
        // Convert to i256 for comparison
        let value_i256 = self.as_i256();

        // Calculate the maximum stored value that can be represented with this precision
        // For precision P, the max stored value is 10^P - 1
        // This is independent of scale - scale only affects how we interpret the value
        let ten = i256::from_i128(10);
        let max_value = ten
            .checked_pow(decimal_type.precision() as _)
            .vortex_expect("precision must exist in i256");
        let min_value = -max_value;

        value_i256 > min_value && value_i256 < max_value
    }

    /// Checked addition. Returns `None` on overflow.
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        checked_widening_binary_op!(self, other, CheckedAdd::checked_add)
    }

    /// Checked subtraction. Returns `None` on overflow.
    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        checked_widening_binary_op!(self, other, CheckedSub::checked_sub)
    }

    /// Checked multiplication. Returns `None` on overflow.
    pub fn checked_mul(&self, other: &Self) -> Option<Self> {
        checked_widening_binary_op!(self, other, CheckedMul::checked_mul)
    }

    /// Checked division. Returns `None` on overflow or division by zero.
    pub fn checked_div(&self, other: &Self) -> Option<Self> {
        checked_widening_binary_op!(self, other, CheckedDiv::checked_div)
    }
}

// Additional trait implementations for decimal types to ensure consistency.

// Comparisons between DecimalValue types should upcast to i256 and operate in the upcast space.
// Decimal values can take on any signed scalar type, but so long as their values are the same
// they are considered the same.
// DecimalScalar handles ensuring that both values being compared have the same precision/scale.
impl PartialEq for DecimalValue {
    fn eq(&self, other: &Self) -> bool {
        let self_upcast = self.as_i256();
        let other_upcast = other.as_i256();

        self_upcast == other_upcast
    }
}

impl Eq for DecimalValue {}

impl PartialOrd for DecimalValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let self_upcast = self.as_i256();
        let other_upcast = other.as_i256();

        self_upcast.partial_cmp(&other_upcast)
    }
}

// Hashing works in the upcast space similar to the other comparison and equality operators.
impl Hash for DecimalValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let self_upcast = self.as_i256();
        self_upcast.hash(state);
    }
}

impl fmt::Display for DecimalValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecimalValue::I8(v8) => write!(f, "decimal8({v8})"),
            DecimalValue::I16(v16) => write!(f, "decimal16({v16})"),
            DecimalValue::I32(v32) => write!(f, "decimal32({v32})"),
            DecimalValue::I64(v64) => write!(f, "decimal64({v64})"),
            DecimalValue::I128(v128) => write!(f, "decimal128({v128})"),
            DecimalValue::I256(v256) => write!(f, "decimal256({v256})"),
        }
    }
}

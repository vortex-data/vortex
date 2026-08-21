// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Evaluation of the sequence equation `A[i] = base + i * multiplier`.
//!
//! All arithmetic stays within 64 bits, kept exact for `u64` values above `i64::MAX` by never
//! routing them through a signed type: unsigned quantities and step magnitudes are handled as
//! `u64`, signed values as `i64`. Construction checks that a sequence is expressible by counting
//! the steps that fit between `base` and the output ptype's boundary rather than by computing the
//! last value in a wider type - see `SequenceData::validate`. The sequence runs monotonically
//! from its first to its last value, so every element of a validated sequence fits the output
//! ptype. Arithmetic modulo `2^bits` agrees with exact arithmetic on values that fit, so kernels
//! materialize the sequence in the output ptype itself, wrapping on overflow - see
//! [`wrapping_value`].

use num_traits::AsPrimitive;
use num_traits::WrappingAdd;
use num_traits::WrappingMul;
use vortex_array::dtype::IntegerPType;
use vortex_array::match_each_pvalue;
use vortex_array::scalar::PValue;

/// An integer type that sequence values can be materialized in.
pub(crate) trait SequenceValue: IntegerPType + WrappingAdd + WrappingMul {
    /// The low bits of a sign-extended `i64`, reinterpreted as `Self`.
    fn wrapping_from_i64(value: i64) -> Self;

    /// The low bits of a `u64`, reinterpreted as `Self`.
    fn wrapping_from_u64(value: u64) -> Self;

    /// The low bits of an index, reinterpreted as `Self`.
    fn wrapping_from_usize(index: usize) -> Self;
}

impl<T> SequenceValue for T
where
    T: IntegerPType + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    #[inline]
    fn wrapping_from_i64(value: i64) -> Self {
        value.as_()
    }

    #[inline]
    fn wrapping_from_u64(value: u64) -> Self {
        value.as_()
    }

    #[inline]
    fn wrapping_from_usize(index: usize) -> Self {
        index.as_()
    }
}

/// `base` and `multiplier` reduced into `O`, the type the sequence's values are computed in.
///
/// `O` is the output ptype in the kernels here. A wider `O` works too - the values then come out
/// correct modulo `2^O::BITS` and have to be truncated to the output ptype, which is what
/// [`SequenceData::wrapping_bits`](crate::SequenceData::wrapping_bits) leaves to its caller.
pub(crate) fn wrapping_parts<O: SequenceValue>(base: PValue, multiplier: PValue) -> Option<(O, O)> {
    Some((wrapping_from(base)?, wrapping_from(multiplier)?))
}

/// The two's-complement bits of an integer [`PValue`], reduced into `O`. `None` for a float.
fn wrapping_from<O: SequenceValue>(value: PValue) -> Option<O> {
    match_each_pvalue!(
        value,
        uint: |v| { Some(O::wrapping_from_u64(v.as_())) },
        int: |v| { Some(O::wrapping_from_i64(v.as_())) },
        float: |_v| { None }
    )
}

/// A step's direction and magnitude: whether it is non-negative, and its absolute value.
///
/// The magnitude is exact even for steps above `i64::MAX`, which stay `u64` throughout.
/// Returns `None` for a float, which no sequence holds.
pub(crate) fn step_parts(step: PValue) -> Option<(bool, u64)> {
    match_each_pvalue!(
        step,
        uint: |v| { Some((true, v.as_())) },
        int: |v| {
            let v: i64 = v.as_();
            Some((v >= 0, v.unsigned_abs()))
        },
        float: |_v| { None }
    )
}

/// The sequence value at `index`, computed modulo `2^bits`.
#[inline]
pub(crate) fn wrapping_value<O: SequenceValue>(base: O, multiplier: O, index: usize) -> O {
    base.wrapping_add(&multiplier.wrapping_mul(&O::wrapping_from_usize(index)))
}

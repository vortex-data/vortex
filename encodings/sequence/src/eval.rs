// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Evaluation of the sequence equation `A[i] = base + i * multiplier`.
//!
//! Arithmetic remains exact within 64 bits by keeping unsigned values out of signed types.
//! Materialization uses wrapping arithmetic in the output ptype.

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

/// Reduces `base` and `multiplier` into the arithmetic type `O`.
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

/// Returns an integer step's direction and magnitude.
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

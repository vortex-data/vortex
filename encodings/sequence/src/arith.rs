// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arithmetic over the values of a [`SequenceData`](crate::SequenceData).
//!
//! Construction validates the first and last value of a sequence against the array's output ptype
//! in exact `i128` space - see [`exact_value`]. A sequence runs monotonically between those two,
//! so every one of its elements fits the output ptype. Arithmetic modulo `2^bits` agrees with
//! exact arithmetic on values that fit, so kernels materialize the sequence in the output ptype
//! itself, wrapping on overflow - see [`wrapping_value`]. Narrow sequences keep narrow arithmetic
//! that way, even when their step is not representable in the output ptype.

use num_traits::AsPrimitive;
use num_traits::WrappingAdd;
use num_traits::WrappingMul;
use num_traits::cast;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_pvalue;
use vortex_array::scalar::PValue;

/// An integer type that sequence values can be materialized in.
pub(crate) trait SequenceValue: IntegerPType + WrappingAdd + WrappingMul {
    /// The low bits of a two's-complement `i128`, reinterpreted as `Self`.
    fn wrapping_from_i128(value: i128) -> Self;

    /// The low bits of an index, reinterpreted as `Self`.
    fn wrapping_from_usize(index: usize) -> Self;
}

impl<T> SequenceValue for T
where
    T: IntegerPType + WrappingAdd + WrappingMul,
    i128: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    #[inline]
    fn wrapping_from_i128(value: i128) -> Self {
        value.as_()
    }

    #[inline]
    fn wrapping_from_usize(index: usize) -> Self {
        index.as_()
    }
}

/// Widens an integer [`PValue`] to `i128`, in which sequence arithmetic is exact.
///
/// Returns `None` for a float, which no sequence holds.
pub(crate) fn widen(value: PValue) -> Option<i128> {
    match_each_pvalue!(
        value,
        uint: |v| { Some(i128::from(v)) },
        int: |v| { Some(i128::from(v)) },
        float: |_v| { None }
    )
}

/// Narrows an exact value into `ptype`.
///
/// Returns `None` if the value is not representable there, or `ptype` is not an integer one.
pub(crate) fn narrow(value: i128, ptype: PType) -> Option<PValue> {
    if !ptype.is_int() {
        return None;
    }

    match_each_integer_ptype!(ptype, |O| { cast::<i128, O>(value).map(PValue::from) })
}

/// The exact value of the sequence element at `index`.
///
/// Returns `None` if the values are not integers, or the result leaves `i128`.
pub(crate) fn exact_value(base: PValue, multiplier: PValue, index: usize) -> Option<i128> {
    let base = widen(base)?;
    let multiplier = widen(multiplier)?;
    let index = i128::try_from(index).ok()?;

    multiplier
        .checked_mul(index)
        .and_then(|offset| base.checked_add(offset))
}

/// `base` and `multiplier` reduced into `O`, the type the sequence's values are computed in.
///
/// `O` is the output ptype in the kernels here. A wider `O` works too - the values then come out
/// correct modulo `2^O::BITS` and have to be truncated to the output ptype, which is what
/// [`SequenceData::wrapping_bits`](crate::SequenceData::wrapping_bits) leaves to its caller.
pub(crate) fn wrapping_parts<O: SequenceValue>(base: PValue, multiplier: PValue) -> Option<(O, O)> {
    Some((
        O::wrapping_from_i128(widen(base)?),
        O::wrapping_from_i128(widen(multiplier)?),
    ))
}

/// The sequence value at `index`, computed modulo `2^bits`.
#[inline]
pub(crate) fn wrapping_value<O: SequenceValue>(base: O, multiplier: O, index: usize) -> O {
    base.wrapping_add(&multiplier.wrapping_mul(&O::wrapping_from_usize(index)))
}

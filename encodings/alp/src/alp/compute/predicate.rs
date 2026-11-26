// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helpers shared by the ALP comparison and between kernels.
//!
//! Both kernels answer a predicate against the encoded integers and then correct the rows that
//! ALP could not encode, whose exact float values live in the patches.

use num_traits::AsPrimitive;
use num_traits::Bounded;
use num_traits::CheckedAdd;
use num_traits::CheckedSub;
use num_traits::One;
use num_traits::PrimInt;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_functions::between::BetweenOptions;
use vortex_functions::between::StrictComparison;
use vortex_functions::builtins::ArrayFunctions;

use crate::ALP;
use crate::ALPArraySlotsExt;
use crate::ALPFloat;
use crate::Exponents;

/// A predicate that has the same answer for every row while keeping the array's validity.
fn constant_predicate(
    array: ArrayView<'_, ALP>,
    value: bool,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let len = array.len();
    Ok(match array.validity()?.union_nullability(nullability) {
        Validity::NonNullable | Validity::AllValid => {
            ConstantArray::new(Scalar::bool(value, nullability), len).into_array()
        }
        Validity::AllInvalid => {
            ConstantArray::new(Scalar::null(DType::Bool(nullability)), len).into_array()
        }
        validity @ Validity::Array(_) => {
            BoolArray::new(BitBuffer::full(value, len), validity).into_array()
        }
    })
}

/// First integer decoding to at least `value`, or strictly above it for a strict bound.
/// `None` denotes the position past the largest integer.
pub(super) fn encoded_bound<F: ALPFloat + NativePType>(
    value: F,
    exponents: Exponents,
    strict: bool,
) -> Option<F::ALPInt> {
    let min = F::ALPInt::min_value();
    let max = F::ALPInt::max_value();
    if !value.is_finite() {
        return value.is_sign_negative().then_some(min);
    }

    let before = |i| {
        let decoded = F::decode_single(i, exponents);
        if strict {
            decoded.is_le(value)
        } else {
            decoded.is_lt(value)
        }
    };
    let guess = F::encode_below(value, exponents);
    let one = F::ALPInt::one();
    let (mut lo, mut hi) = if before(guess) {
        let next = guess.checked_add(&one)?;
        if !before(next) {
            return Some(next);
        }
        (next, max)
    } else {
        let Some(prev) = guess.checked_sub(&one) else {
            return Some(min);
        };
        if before(prev) {
            return Some(guess);
        }
        (min, prev)
    };

    // Decode is monotone, but rounding can give many adjacent integers the same float.
    while lo < hi {
        let mid = (lo & hi) + ((lo ^ hi) >> 1); // floor midpoint without signed overflow
        if before(mid) {
            lo = mid + one;
        } else {
            hi = mid;
        }
    }
    (!before(lo)).then_some(lo)
}

/// Tests membership in the half-open encoded interval. Either end can be past the maximum.
pub(super) fn encoded_range<I: PrimInt + NativePType + Into<PValue>>(
    array: ArrayView<'_, ALP>,
    start: Option<I>,
    end: Option<I>,
    negate: bool,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let Some(start) = start else {
        return constant_predicate(array, negate, nullability);
    };
    let scalar =
        |i: I| ConstantArray::new(Scalar::primitive(i, nullability), array.len()).into_array();
    let ints = array.encoded();
    let compare =
        |i, op: CompareOperator| ints.compare(scalar(i), if negate { op.inverse() } else { op });
    match end {
        Some(end) if end <= start => constant_predicate(array, negate, nullability),
        Some(end) if end == start + I::one() => compare(start, CompareOperator::Eq),
        Some(end) if start == I::min_value() => compare(end, CompareOperator::Lt),
        Some(end) => {
            let result = ints.clone().between(
                scalar(start),
                scalar(end),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::Strict,
                },
            )?;
            if negate { result.not() } else { Ok(result) }
        }
        None if start == I::min_value() => constant_predicate(array, !negate, nullability),
        None => compare(start, CompareOperator::Gte),
    }
}

/// Re-evaluates `predicate` on each patched row of `result`, a boolean answer computed from the
/// encoded values alone.
pub(super) fn apply_patch_predicate<F: NativePType>(
    result: ArrayRef,
    patches: &Patches,
    predicate: impl Fn(F) -> bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let result = result.execute::<BoolArray>(ctx)?;
    let validity = result.validity()?;
    let mut bits = result
        .into_bit_buffer()
        .try_into_mut()
        .unwrap_or_else(|bits| BitBufferMut::copy_from(&bits));

    let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
    let values = values.as_slice::<F>();
    let offset = patches.offset();

    match_each_integer_ptype!(indices.ptype(), |I| {
        set_patched_bits(
            &mut bits,
            indices.as_slice::<I>(),
            values,
            offset,
            predicate,
        )
    });

    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}

/// Sets each patched row of `bits` that lies within its range to `predicate` of its patch value.
fn set_patched_bits<I: AsPrimitive<usize>, F: NativePType>(
    bits: &mut BitBufferMut,
    indices: &[I],
    values: &[F],
    offset: usize,
    predicate: impl Fn(F) -> bool,
) {
    for (&index, &value) in indices.iter().zip(values) {
        let index: usize = index.as_();
        // Patch indices are absolute; a slice keeps only those within its range.
        if let Some(local) = index.checked_sub(offset)
            && local < bits.len()
        {
            bits.set_to(local, predicate(value));
        }
    }
}

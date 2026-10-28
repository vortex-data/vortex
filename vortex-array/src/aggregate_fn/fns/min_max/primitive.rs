// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Either;
use num_traits::Float;
use vortex_buffer::BitBuffer;
use vortex_compute::lane_kernels::IndexedFoldExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::Nullability::NonNullable;
use crate::match_each_native_ptype;
use crate::scalar::PValue;
use crate::scalar::Scalar;

pub(super) fn accumulate_primitive(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    p: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let mask = p.as_ref().validity()?.execute_mask(p.as_ref().len(), ctx)?;
    let valid = match &mask {
        Mask::AllFalse(_) => return Ok(()),
        Mask::AllTrue(_) => None,
        Mask::Values(v) => Some(v.bit_buffer()),
    };

    let local = match_each_native_ptype!(p.ptype(),
        integral: |T| { integer_min_max(p.as_slice::<T>(), valid).map(min_max_result) },
        floating: |T| {
            float_min_max(p.as_slice::<T>(), valid, args.options.skip_nans).map(min_max_result)
        }
    );
    partial.merge(args, local);
    Ok(())
}

/// Min/max of the valid values of an integer slice.
///
/// Kept out of line so each integer type gets a function of its own, as the float types do: inlined
/// into the dispatch over every type, the scalar 64-bit loop of baseline x86-64 ran 40% slower.
#[inline(never)]
fn integer_min_max<T>(slice: &[T], valid: Option<&BitBuffer>) -> Option<(T, T)>
where
    T: NativePType,
{
    match valid {
        None => integer_min_max_raw(slice),
        Some(valid) => {
            let first = valid
                .set_indices()
                .next()
                .vortex_expect("a mask with values has a valid row");
            Some(integer_min_max_masked(slice, valid, slice[first]))
        }
    }
}

/// Min/max of an all-valid integer slice as native values. Autovectorizes to packed min/max.
fn integer_min_max_raw<T>(slice: &[T]) -> Option<(T, T)>
where
    T: NativePType,
{
    let (&first, rest) = slice.split_first()?;
    let mut min = first;
    let mut max = first;
    for &v in rest {
        if v.is_lt(min) {
            min = v;
        }
        if v.is_gt(max) {
            max = v;
        }
    }
    Some((min, max))
}

/// Min/max of the valid values of an integer slice, reading null lanes as `valid_value`, a value of
/// some valid lane, which leaves both extrema unchanged. Selecting per lane rather than walking the
/// runs of valid values keeps the packed min/max of [`integer_min_max_raw`] however short the runs.
fn integer_min_max_masked<T>(slice: &[T], valid: &BitBuffer, valid_value: T) -> (T, T)
where
    T: NativePType,
{
    slice.fold_masked(
        valid,
        valid_value,
        (valid_value, valid_value),
        |(min, max), v| {
            (
                if v.is_lt(min) { v } else { min },
                if v.is_gt(max) { v } else { max },
            )
        },
    )
}

/// Min/max of the valid values of a float slice under the total order, or `None` if there are
/// none.
///
/// With `skip_nans`, NaN values are ignored. Otherwise a valid NaN makes both extrema NaN, which
/// poisons the partial state.
///
/// The extrema are folded with float comparisons, which every target compares in vector registers,
/// unlike the 64-bit integer keys of the total order, which baseline x86-64 cannot. A comparison
/// with a NaN is false, so the folds skip NaN values for free, and null lanes read `seed`, a valid
/// non-NaN value, which leaves both extrema unchanged. Each extremum is a fold of its own: a fold
/// into one float per lane vectorizes on every target, where a fold into a pair of floats can stay
/// scalar. Two cases the comparisons cannot see are settled after the folds, each with another
/// pass that runs only when needed:
///
/// - Without `skip_nans`, a third fold sums `v - v`, which turns NaN only for a NaN or infinite
///   value. When it does, [`find_nan`] looks for a valid NaN.
/// - The total order puts `-0.0` below `0.0`, but the comparisons see them as equal. When an
///   extremum is zero, [`has_zero_with_sign`] finds which signs are present.
fn float_min_max<T>(values: &[T], valid: Option<&BitBuffer>, skip_nans: bool) -> Option<(T, T)>
where
    T: NativePType + Float,
{
    let mut indices = valid_indices(values.len(), valid);
    let seed = if skip_nans {
        indices.map(|i| values[i]).find(|&v| !Float::is_nan(v))?
    } else {
        values[indices.next()?]
    };
    if Float::is_nan(seed) {
        return Some((seed, seed));
    }

    if !skip_nans {
        let non_finite = fold_floats(values, valid, seed, T::zero(), |e, v| e + (v - v), add);
        if Float::is_nan(non_finite)
            && let Some(nan) = find_nan(values, valid)
        {
            return Some((nan, nan));
        }
    }

    let min = fold_floats(values, valid, seed, seed, lower, lower);
    let max = fold_floats(values, valid, seed, seed, higher, higher);

    let min = if min.is_zero() {
        zero_with_sign(has_zero_with_sign(values, valid, true))
    } else {
        min
    };
    let max = if max.is_zero() {
        zero_with_sign(!has_zero_with_sign(values, valid, false))
    } else {
        max
    };
    Some((min, max))
}

/// Fold the valid float values with [`IndexedFoldExt::fold_lanes`], or its masked form reading null
/// lanes as `null`, with as many lanes as fill 64 bytes per accumulator.
#[inline]
fn fold_floats<T, A>(
    values: &[T],
    valid: Option<&BitBuffer>,
    null: T,
    init: A,
    f: impl Fn(A, T) -> A,
    merge: impl Fn(A, A) -> A,
) -> A
where
    T: NativePType,
    A: Copy,
{
    macro_rules! fold {
        ($lanes:literal) => {
            match valid {
                None => values.fold_lanes::<$lanes, _, _, _>(init, f, merge),
                Some(valid) => values.fold_lanes_masked::<$lanes, _, _, _>(valid, null, init, f, merge),
            }
        };
    }

    match size_of::<T>() {
        2 => fold!(32),
        4 => fold!(16),
        _ => fold!(8),
    }
}

fn add<T: Float>(a: T, b: T) -> T {
    a + b
}

/// `v` if it compares below `min`, else `min`. NaN never compares below.
#[inline]
fn lower<T: Float>(min: T, v: T) -> T {
    if v < min { v } else { min }
}

/// `v` if it compares above `max`, else `max`. NaN never compares above.
#[inline]
fn higher<T: Float>(max: T, v: T) -> T {
    if v > max { v } else { max }
}

/// The first valid NaN.
#[cold]
#[inline(never)]
fn find_nan<T: Float>(values: &[T], valid: Option<&BitBuffer>) -> Option<T> {
    valid_indices(values.len(), valid)
        .map(|i| values[i])
        .find(|v| v.is_nan())
}

/// Whether a valid value is a zero of the given sign.
#[cold]
#[inline(never)]
fn has_zero_with_sign<T: Float>(values: &[T], valid: Option<&BitBuffer>, negative: bool) -> bool {
    valid_indices(values.len(), valid)
        .any(|i| values[i].is_zero() && values[i].is_sign_negative() == negative)
}

fn zero_with_sign<T: Float>(negative: bool) -> T {
    if negative { -T::zero() } else { T::zero() }
}

fn valid_indices(len: usize, valid: Option<&BitBuffer>) -> impl Iterator<Item = usize> + '_ {
    match valid {
        None => Either::Left(0..len),
        Some(valid) => Either::Right(valid.set_indices()),
    }
}

fn min_max_result<T>((min, max): (T, T)) -> MinMaxResult
where
    T: NativePType,
    PValue: From<T>,
{
    MinMaxResult {
        min: Scalar::primitive(min, NonNullable),
        max: Scalar::primitive(max, NonNullable),
    }
}

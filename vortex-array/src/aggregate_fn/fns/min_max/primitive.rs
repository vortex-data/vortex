// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::Bounded;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::Nullability::NonNullable;
use crate::match_each_native_ptype;
use crate::scalar::PValue;
use crate::scalar::Scalar;

pub(super) fn accumulate_primitive(
    partial: &mut MinMaxPartial,
    p: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match_each_native_ptype!(p.ptype(), |T| {
        let local = compute_min_max_with_validity::<T>(p, ctx)?;
        partial.merge(local);
        Ok(())
    })
}

fn compute_min_max_with_validity<T>(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<MinMaxResult>>
where
    T: NativePType + PartialOrd + Bounded,
    PValue: From<T>,
{
    let values = array.as_slice::<T>();
    let (vmin, vmax) = match array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?
    {
        Mask::AllTrue(_) => minmax_all(values),
        Mask::AllFalse(_) => return Ok(None),
        Mask::Values(v) => minmax_masked(values, v.bit_buffer()),
    };
    // Neutral seeds invert (`vmin > vmax`) exactly when no non-NaN value was folded, so this both
    // detects the empty/all-NaN case and produces the result otherwise.
    Ok((vmin <= vmax).then(|| MinMaxResult {
        min: Scalar::primitive(vmin, NonNullable),
        max: Scalar::primitive(vmax, NonNullable),
    }))
}

/// Plain-comparison min/max over all values, skipping NaN (NaN fails both comparisons). For every
/// non-NaN input this matches the previous `total_compare`-based result; `±0.0` ties are
/// numerically equal and not distinguished, which is irrelevant to callers (range pruning, casts).
#[multiversion::multiversion(targets("x86_64+avx512f", "x86_64+avx2", "aarch64+neon"))]
fn minmax_all<T: NativePType + PartialOrd + Bounded>(values: &[T]) -> (T, T) {
    let mut vmin = T::max_value();
    let mut vmax = T::min_value();
    for &v in values {
        if v < vmin {
            vmin = v;
        }
        if v > vmax {
            vmax = v;
        }
    }
    (vmin, vmax)
}

/// Validity-gated min/max, branch-free: invalid lanes fold against neutral bounds (never winning),
/// NaN is skipped. Word-chunked so it vectorizes regardless of null density.
#[multiversion::multiversion(targets("x86_64+avx512f", "x86_64+avx2", "aarch64+neon"))]
fn minmax_masked<T: NativePType + PartialOrd + Bounded>(
    values: &[T],
    validity: &BitBuffer,
) -> (T, T) {
    let hi = T::max_value();
    let lo = T::min_value();
    let mut vmin = hi;
    let mut vmax = lo;
    let chunks = validity.chunks();
    let mut base = 0usize;
    for word in chunks.iter() {
        for (j, &v) in values[base..base + 64].iter().enumerate() {
            let valid = (word >> j) & 1 != 0;
            let for_min = if valid { v } else { hi };
            let for_max = if valid { v } else { lo };
            if for_min < vmin {
                vmin = for_min;
            }
            if for_max > vmax {
                vmax = for_max;
            }
        }
        base += 64;
    }
    let remainder = chunks.remainder_bits();
    for (j, &v) in values[base..].iter().enumerate() {
        let valid = (remainder >> j) & 1 != 0;
        let for_min = if valid { v } else { hi };
        let for_max = if valid { v } else { lo };
        if for_min < vmin {
            vmin = for_min;
        }
        if for_max > vmax {
            vmax = for_max;
        }
    }
    (vmin, vmax)
}

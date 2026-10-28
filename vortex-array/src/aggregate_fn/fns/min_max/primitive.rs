// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use half::f16;
use vortex_buffer::BitBuffer;
use vortex_compute::lane_kernels::IndexedFoldExt;
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
    let skip_nans = args.options.skip_nans;
    let validity = p.as_ref().validity()?.execute_mask(p.as_ref().len(), ctx)?;
    let valid = match &validity {
        Mask::AllTrue(_) => None,
        Mask::AllFalse(_) => return Ok(()),
        Mask::Values(v) => Some(v.bit_buffer()),
    };
    let local = match_each_native_ptype!(p.ptype(),
        integral: |T| { integer_min_max(p.as_slice::<T>(), valid).map(min_max_result) },
        floating: |T| { float_min_max(p.as_slice::<T>(), valid, skip_nans).map(min_max_result) }
    );
    partial.merge(args, local);
    Ok(())
}

/// Min/max of the valid values of an integer slice, given its validity unless every value is
/// valid.
///
/// Null lanes read as a valid value, which leaves both extrema unchanged, so the reduction
/// autovectorizes to packed min/max however the valid values are spread.
fn integer_min_max<T: NativePType>(slice: &[T], valid: Option<&BitBuffer>) -> Option<(T, T)> {
    let first = slice[first_valid(slice.len(), valid, |_| true)?];
    let min_max = |(min, max): (T, T), v: T| {
        (
            if v.is_lt(min) { v } else { min },
            if v.is_gt(max) { v } else { max },
        )
    };
    Some(match valid {
        None => slice.iter().copied().fold((first, first), min_max),
        Some(valid) => slice.fold_masked(valid, first, (first, first), min_max),
    })
}

/// Min/max of the valid values of a float slice under the IEEE 754 total order, skipping NaN
/// values when `skip_nans` is set.
///
/// Without `skip_nans`, NaN values sort to the ends of the total order, so a NaN extremum lets the
/// partial's merge poison the result. The reduction runs over [`TotalOrderKey`]s, integers that
/// order exactly as the floats do, so it autovectorizes to packed integer min/max; null and
/// skipped lanes read as the key of a value that is neither, which leaves both extrema unchanged.
fn float_min_max<T: TotalOrderKey>(
    slice: &[T],
    valid: Option<&BitBuffer>,
    skip_nans: bool,
) -> Option<(T, T)> {
    let first = slice[first_valid(slice.len(), valid, |i| !(skip_nans && slice[i].is_nan()))?];
    let first_key = first.key();
    let min_max = |(min, max): (T::Key, T::Key), v: T| {
        let key = if skip_nans && v.is_nan() {
            first_key
        } else {
            v.key()
        };
        (min.min(key), max.max(key))
    };
    let (min, max) = match valid {
        None => slice.iter().copied().fold((first_key, first_key), min_max),
        Some(valid) => slice.fold_masked(valid, first, (first_key, first_key), min_max),
    };
    Some((T::from_key(min), T::from_key(max)))
}

/// The index of the first valid lane for which `include` holds, or `None` if there is none.
fn first_valid(
    len: usize,
    valid: Option<&BitBuffer>,
    include: impl Fn(usize) -> bool,
) -> Option<usize> {
    match valid {
        None => (0..len).find(|&i| include(i)),
        Some(valid) => valid.set_indices().find(|&i| include(i)),
    }
}

/// A float whose IEEE 754 total order, the order of `total_cmp`, is the order of a signed
/// integer key derived from its bits.
///
/// Flipping every bit but the sign of a negative float's bits reverses the order of the negative
/// floats, so the bits order as `total_cmp` does when read as a signed integer. This is how
/// `total_cmp` itself is implemented. The transform keeps the sign bit, so it is its own inverse.
trait TotalOrderKey: NativePType {
    type Key: Copy + Ord;

    fn key(self) -> Self::Key;

    fn from_key(key: Self::Key) -> Self;
}

macro_rules! total_order_key {
    ($float:ty, $signed:ty, $unsigned:ty) => {
        impl TotalOrderKey for $float {
            type Key = $signed;

            #[inline]
            #[expect(
                clippy::cast_possible_wrap,
                reason = "the key reinterprets the float's bits"
            )]
            fn key(self) -> $signed {
                let bits = self.to_bits() as $signed;
                bits ^ ((((bits >> (<$signed>::BITS - 1)) as $unsigned) >> 1) as $signed)
            }

            #[inline]
            #[expect(
                clippy::cast_sign_loss,
                reason = "the key reinterprets the float's bits"
            )]
            fn from_key(key: $signed) -> Self {
                let bits = key ^ ((((key >> (<$signed>::BITS - 1)) as $unsigned) >> 1) as $signed);
                <$float>::from_bits(bits as $unsigned)
            }
        }
    };
}

total_order_key!(f16, i16, u16);
total_order_key!(f32, i32, u32);
total_order_key!(f64, i64, u64);

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

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        reason = "narrows the f64 cases to f32"
    )]

    use half::f16;
    use rstest::rstest;

    use super::TotalOrderKey;

    /// Keys order exactly as `total_cmp` does, and convert back to the same bits.
    #[rstest]
    #[case(&[f64::NAN, -f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0, 1.5, -1.5, f64::MIN_POSITIVE, -f64::MIN_POSITIVE, f64::MAX, f64::MIN])]
    fn total_order_keys_order_as_total_cmp(#[case] values: &[f64]) {
        for &a in values {
            assert_eq!(f64::from_key(a.key()).to_bits(), a.to_bits());
            assert_eq!(
                f32::from_key((a as f32).key()).to_bits(),
                (a as f32).to_bits()
            );
            let h = f16::from_f64(a);
            assert_eq!(f16::from_key(h.key()).to_bits(), h.to_bits());
            for &b in values {
                assert_eq!(a.key().cmp(&b.key()), a.total_cmp(&b), "{a} vs {b}");
                let (af, bf) = (a as f32, b as f32);
                assert_eq!(af.key().cmp(&bf.key()), af.total_cmp(&bf), "{af} vs {bf}");
            }
        }
    }
}

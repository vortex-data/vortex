// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::Bounded;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn;
use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::match_each_native_ptype;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::validity::Validity;

impl CastReduce for Primitive {
    fn cast(array: ArrayView<'_, Primitive>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Only the same ptype is reducible without execution; type changes need the kernel
        // to verify values fit in the target range.
        let DType::Primitive(new_ptype, new_nullability) = dtype else {
            return Ok(None);
        };
        if *new_ptype != array.ptype() {
            return Ok(None);
        }

        let Some(new_validity) = array
            .validity()?
            .trivially_cast_nullability(*new_nullability, array.len())?
        else {
            return Ok(None);
        };

        // SAFETY: validity and data buffer still have same length.
        Ok(Some(unsafe {
            PrimitiveArray::new_unchecked_from_handle(
                array.buffer_handle().clone(),
                array.ptype(),
                new_validity,
            )
            .into_array()
        }))
    }
}

impl CastKernel for Primitive {
    fn cast(
        array: ArrayView<'_, Primitive>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let DType::Primitive(new_ptype, new_nullability) = dtype else {
            return Ok(None);
        };
        let (new_ptype, new_nullability) = (*new_ptype, *new_nullability);
        let src_ptype = array.ptype();

        let new_validity = array
            .validity()?
            .cast_nullability(new_nullability, array.len(), ctx)?;

        // Same bit representation: either the same ptype (only the nullability changed) or two
        // same-width integers (identical layout under 2's complement). The only non-trivial case
        // is the sign change between same-width ints, which still needs a value-range check.
        let same_rep = src_ptype == new_ptype
            || (src_ptype.is_int()
                && new_ptype.is_int()
                && src_ptype.byte_width() == new_ptype.byte_width());
        if same_rep {
            if !values_fit_in(array, new_ptype, ctx, true) {
                vortex_bail!(
                    Compute: "Cannot cast {} to {} — values exceed target range",
                    src_ptype, new_ptype,
                );
            }
            return Ok(Some(reinterpret(array, new_ptype, new_validity)));
        }

        // Different bit rep: cast each element. `cast_values` picks a pure or checked loop based
        // on whether the conversion is statically infallible.
        Ok(Some(match_each_native_ptype!(new_ptype, |T| {
            match_each_native_ptype!(src_ptype, |F| {
                cast_values::<F, T>(array, new_validity, ctx)?
            })
        })))
    }
}

/// Cast values from `F` to `T`. For infallible casts this is a pure pass; for fallible casts the
/// kernel fuses the conversion with a validity-aware min/max reduction over the source and bails
/// if those bounds don't fit `T`.
fn cast_values<F, T>(
    array: ArrayView<'_, Primitive>,
    new_validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    F: NativePType + AsPrimitive<T> + PartialOrd + Bounded + Into<PValue>,
    T: NativePType,
{
    let values = array.as_slice::<F>();

    // Fast path: statically infallible, or cached min/max prove every valid value fits in `T`.
    // The cached check never triggers a stats computation — if the bounds aren't already known
    // we fall through to the fused kernel below.
    if values_always_fit(F::PTYPE, T::PTYPE) || values_fit_in(array, T::PTYPE, ctx, false) {
        return Ok(PrimitiveArray::new(cast::<F, T>(values), new_validity).into_array());
    }

    // TODO(joe): if the values source and target have the same bit-width we can
    // mutate in place.

    // Fallible path. A value can only exceed `T`'s range if it is strictly below the smallest or
    // above the largest *valid* value, so a single validity-aware min/max over the source fully
    // decides whether the cast can overflow. We fuse that reduction with the value conversion in
    // one SIMD-friendly pass: every lane is cast unconditionally (out-of-range values at invalid
    // positions wrap/saturate harmlessly under `as` and are masked out), while the running min/max
    // folds only valid, non-NaN lanes. If the resulting bounds don't fit `T`, the cast bails.
    let len = values.len();
    let mask = array.validity()?.execute_mask(len, ctx)?;

    let mut out = BufferMut::<T>::with_capacity(len);
    // SAFETY: `T` is a primitive numeric type for which every bit pattern is valid, and every
    // element of `dst` is written exactly once below before the buffer is frozen and read.
    unsafe { out.set_len(len) };
    let dst = out.as_mut_slice();

    let bounds = match &mask {
        Mask::AllTrue(_) => cast_and_bounds_all::<F, T>(values, dst),
        Mask::AllFalse(_) => {
            dst.fill(T::default());
            None
        }
        Mask::Values(m) => cast_and_bounds_masked::<F, T>(values, m.bit_buffer(), dst),
    };

    if let Some((min, max)) = bounds {
        let target = DType::Primitive(T::PTYPE, Nullability::NonNullable);
        let fits = |v: F| {
            Scalar::primitive(v, Nullability::NonNullable)
                .cast(&target)
                .is_ok()
        };
        if !fits(min) || !fits(max) {
            vortex_bail!(
                Compute: "Cannot cast {} to {} — value exceeds target range",
                F::PTYPE, T::PTYPE,
            );
        }
    }

    Ok(PrimitiveArray::new(out.freeze(), new_validity).into_array())
}

/// Fused cast + min/max for an all-valid run. Writes `values as T` into `dst` and returns the
/// numeric min/max of `values`, ignoring NaN. Returns `None` when there is no non-NaN value (so
/// the caller skips the range check, since nothing can be out of range).
#[multiversion::multiversion(targets("x86_64+avx512f", "x86_64+avx2", "aarch64+neon"))]
fn cast_and_bounds_all<F, T>(values: &[F], dst: &mut [T]) -> Option<(F, F)>
where
    F: NativePType + AsPrimitive<T> + PartialOrd + Bounded,
    T: NativePType,
{
    let mut vmin = F::max_value();
    let mut vmax = F::min_value();
    for (out, &v) in dst.iter_mut().zip(values) {
        *out = v.as_();
        // NaN fails both comparisons and is therefore skipped, matching `min_max` semantics.
        if v < vmin {
            vmin = v;
        }
        if v > vmax {
            vmax = v;
        }
    }
    // `vmin <= vmax` holds exactly when at least one non-NaN value was folded.
    (vmin <= vmax).then_some((vmin, vmax))
}

/// Fused cast + validity-aware min/max. Writes `values as T` into `dst` for every lane, but folds
/// only lanes whose validity bit is set into the returned min/max (NaN excluded). Invalid lanes
/// may hold out-of-range values; those are masked out and never affect the bounds.
///
/// The lane gating is branch-free (invalid lanes are folded against neutral bounds rather than
/// skipped) so the loop vectorizes regardless of null density — a data-dependent branch on the
/// validity word would otherwise serialize the reduction whenever nulls are present.
#[multiversion::multiversion(targets("x86_64+avx512f", "x86_64+avx2", "aarch64+neon"))]
fn cast_and_bounds_masked<F, T>(values: &[F], validity: &BitBuffer, dst: &mut [T]) -> Option<(F, F)>
where
    F: NativePType + AsPrimitive<T> + PartialOrd + Bounded,
    T: NativePType,
{
    // Invalid lanes fold against these neutrals, which can never win the min/max. A valid NaN
    // fails both comparisons and is skipped, matching `min_max`'s NaN filtering.
    let hi_neutral = F::max_value();
    let lo_neutral = F::min_value();
    let mut vmin = hi_neutral;
    let mut vmax = lo_neutral;

    let chunks = validity.chunks();
    let mut base = 0usize;
    for word in chunks.iter() {
        let vblk = &values[base..base + 64];
        let oblk = &mut dst[base..base + 64];
        for (j, (out, &v)) in oblk.iter_mut().zip(vblk).enumerate() {
            *out = v.as_();
            let valid = (word >> j) & 1 != 0;
            let for_min = if valid { v } else { hi_neutral };
            let for_max = if valid { v } else { lo_neutral };
            if for_min < vmin {
                vmin = for_min;
            }
            if for_max > vmax {
                vmax = for_max;
            }
        }
        base += 64;
    }

    // Trailing lanes that don't fill a 64-bit chunk.
    let remainder = chunks.remainder_bits();
    for (j, (&v, out)) in values[base..]
        .iter()
        .zip(dst[base..].iter_mut())
        .enumerate()
    {
        *out = v.as_();
        let valid = (remainder >> j) & 1 != 0;
        let for_min = if valid { v } else { hi_neutral };
        let for_max = if valid { v } else { lo_neutral };
        if for_min < vmin {
            vmin = for_min;
        }
        if for_max > vmax {
            vmax = for_max;
        }
    }

    (vmin <= vmax).then_some((vmin, vmax))
}

/// Out-of-range values at invalid positions are truncated/wrapped by `as`, which is fine because
/// they are masked out by validity.
fn cast<F: NativePType + AsPrimitive<T>, T: NativePType>(array: &[F]) -> Buffer<T> {
    BufferMut::from_trusted_len_iter(array.iter().map(|&src| src.as_())).freeze()
}

fn reinterpret(
    array: ArrayView<'_, Primitive>,
    new_ptype: PType,
    new_validity: Validity,
) -> ArrayRef {
    // SAFETY: caller has verified the bit representation is compatible and that validity length
    // still matches the buffer length.
    unsafe {
        PrimitiveArray::new_unchecked_from_handle(
            array.buffer_handle().clone(),
            new_ptype,
            new_validity,
        )
    }
    .into_array()
}

/// Returns `true` if every value of `src` is guaranteed representable in `target` without
/// overflow. Precision may be lost (e.g. large integers cast to `f32`), but the cast can never
/// produce an out-of-range result.
fn values_always_fit(src: PType, target: PType) -> bool {
    if src == target {
        return true;
    }
    if src.is_int() && target.is_int() {
        return target.byte_width() > src.byte_width()
            && (src.is_unsigned_int() || target.is_signed_int());
    }
    if src.is_float() && target.is_float() {
        return target.byte_width() > src.byte_width();
    }
    src.is_int() && matches!(target, PType::F32 | PType::F64)
}

/// Returns `true` if all valid values in `array` are representable as `target_ptype`.
///
/// Cached min/max statistics are consulted first. If either bound is missing, the function either
/// computes them with a single pass (when `compute` is `true`) or returns `false` so the caller
/// can fall back to a slower path (when `compute` is `false`).
fn values_fit_in(
    array: ArrayView<'_, Primitive>,
    target_ptype: PType,
    ctx: &mut ExecutionCtx,
    compute: bool,
) -> bool {
    let target_dtype = DType::Primitive(target_ptype, Nullability::NonNullable);
    if let Some(fits) = cached_values_fit_in(array, &target_dtype) {
        return fits;
    }
    if !compute {
        return false;
    }
    aggregate_fn::fns::min_max::min_max(array.array(), ctx)
        .ok()
        .flatten()
        .is_none_or(|mm| mm.min.cast(&target_dtype).is_ok() && mm.max.cast(&target_dtype).is_ok())
}

/// Cached-only check: returns `Some(fits)` if both `Min` and `Max` are present as `Exact` in the
/// stats cache, otherwise `None`.
fn cached_values_fit_in(array: ArrayView<'_, Primitive>, target_dtype: &DType) -> Option<bool> {
    let stats = array.array().statistics();
    let min = stats.get(Stat::Min).as_exact()?;
    let max = stats.get(Stat::Max).as_exact()?;
    Some(min.cast(target_dtype).is_ok() && max.cast(target_dtype).is_ok())
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexError;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    #[test]
    fn cast_u32_u8() {
        let arr = buffer![0u32, 10, 200].into_array();

        // cast from u32 to u8
        #[expect(deprecated)]
        let p = arr.cast(PType::U8.into()).unwrap().to_primitive();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([0u8, 10, 200]));
        assert!(matches!(p.validity(), Ok(Validity::NonNullable)));

        // to nullable
        #[expect(deprecated)]
        let p = p
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::Nullable))
            .unwrap()
            .to_primitive();
        assert_arrays_eq!(
            p,
            PrimitiveArray::new(buffer![0u8, 10, 200], Validity::AllValid)
        );
        assert!(matches!(p.validity(), Ok(Validity::AllValid)));

        // back to non-nullable
        #[expect(deprecated)]
        let p = p
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::NonNullable))
            .unwrap()
            .to_primitive();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([0u8, 10, 200]));
        assert!(matches!(p.validity(), Ok(Validity::NonNullable)));

        // to nullable u32
        #[expect(deprecated)]
        let p = p
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap()
            .to_primitive();
        assert_arrays_eq!(
            p,
            PrimitiveArray::new(buffer![0u32, 10, 200], Validity::AllValid)
        );
        assert!(matches!(p.validity(), Ok(Validity::AllValid)));

        // to non-nullable u8
        #[expect(deprecated)]
        let p = p
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::NonNullable))
            .unwrap()
            .to_primitive();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([0u8, 10, 200]));
        assert!(matches!(p.validity(), Ok(Validity::NonNullable)));
    }

    #[test]
    fn cast_u32_f32() {
        let arr = buffer![0u32, 10, 200].into_array();
        #[expect(deprecated)]
        let u8arr = arr.cast(PType::F32.into()).unwrap().to_primitive();
        assert_arrays_eq!(u8arr, PrimitiveArray::from_iter([0.0f32, 10., 200.]));
    }

    #[test]
    fn cast_i32_u32() {
        let arr = buffer![-1i32].into_array();
        #[expect(deprecated)]
        let error = arr
            .cast(PType::U32.into())
            .and_then(|a| a.to_canonical().map(|c| c.into_array()))
            .unwrap_err();
        assert!(matches!(error, VortexError::Compute(..)));
        assert!(error.to_string().contains("values exceed target range"));
    }

    #[test]
    fn cast_array_with_nulls_to_nonnullable() {
        let arr = PrimitiveArray::from_option_iter([Some(-1i32), None, Some(10)]);
        #[expect(deprecated)]
        let err = arr
            .into_array()
            .cast(PType::I32.into())
            .and_then(|a| a.to_canonical().map(|c| c.into_array()))
            .unwrap_err();

        assert!(matches!(err, VortexError::InvalidArgument(..)));
        assert!(
            err.to_string()
                .contains("Cannot cast array with invalid values to non-nullable type.")
        );
    }

    #[test]
    fn cast_with_invalid_nulls() {
        let arr = PrimitiveArray::new(
            buffer![-1i32, 0, 10],
            Validity::from_iter([false, true, true]),
        );
        #[expect(deprecated)]
        let p = arr
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap()
            .to_primitive();
        assert_arrays_eq!(
            p,
            PrimitiveArray::from_option_iter([None, Some(0u32), Some(10)])
        );
        assert_eq!(
            p.as_ref()
                .validity()
                .unwrap()
                .execute_mask(p.as_ref().len(), &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Mask::from(BitBuffer::from(vec![false, true, true]))
        );
    }

    /// Same-width integer cast where all values fit: should reinterpret the
    /// buffer without allocation (pointer identity).
    #[test]
    fn cast_same_width_int_reinterprets_buffer() -> vortex_error::VortexResult<()> {
        let src = PrimitiveArray::from_iter([0u32, 10, 100]);
        let src_ptr = src.as_slice::<u32>().as_ptr();

        #[expect(deprecated)]
        let dst = src.into_array().cast(PType::I32.into())?.to_primitive();
        let dst_ptr = dst.as_slice::<i32>().as_ptr();

        // Zero-copy: the data pointer should be identical.
        assert_eq!(src_ptr as usize, dst_ptr as usize);
        assert_arrays_eq!(dst, PrimitiveArray::from_iter([0i32, 10, 100]));
        Ok(())
    }

    /// Same-width integer cast where values don't fit: should fall through
    /// to the allocating path and produce an error.
    #[test]
    fn cast_same_width_int_out_of_range_errors() {
        let arr = buffer![u32::MAX].into_array();
        #[expect(deprecated)]
        let err = arr
            .cast(PType::I32.into())
            .and_then(|a| a.to_canonical().map(|c| c.into_array()))
            .unwrap_err();
        assert!(matches!(err, VortexError::Compute(..)));
    }

    /// All-null array cast between same-width types should succeed without
    /// touching the buffer contents.
    #[test]
    fn cast_same_width_all_null() -> vortex_error::VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![0xFFu8, 0xFF], Validity::AllInvalid);
        #[expect(deprecated)]
        let casted = arr
            .into_array()
            .cast(DType::Primitive(PType::I8, Nullability::Nullable))?
            .to_primitive();
        assert_eq!(casted.len(), 2);
        assert!(matches!(casted.validity(), Ok(Validity::AllInvalid)));
        Ok(())
    }

    /// Same-width integer cast with nullable values: out-of-range nulls should
    /// not prevent the cast from succeeding.
    #[test]
    fn cast_same_width_int_nullable_with_out_of_range_nulls() -> vortex_error::VortexResult<()> {
        // The null position holds u32::MAX which doesn't fit in i32, but it's
        // masked as invalid so the cast should still succeed via reinterpret.
        let arr = PrimitiveArray::new(
            buffer![u32::MAX, 0u32, 42u32],
            Validity::from_iter([false, true, true]),
        );
        #[expect(deprecated)]
        let casted = arr
            .into_array()
            .cast(DType::Primitive(PType::I32, Nullability::Nullable))?
            .to_primitive();
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_option_iter([None, Some(0i32), Some(42)])
        );
        Ok(())
    }

    #[test]
    fn cast_u32_to_u8_with_out_of_range_nulls() -> vortex_error::VortexResult<()> {
        let arr = PrimitiveArray::new(
            buffer![1000u32, 10u32, 42u32],
            Validity::from_iter([false, true, true]),
        );
        #[expect(deprecated)]
        let casted = arr
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::Nullable))?
            .to_primitive();
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_option_iter([None, Some(10u8), Some(42)])
        );
        Ok(())
    }

    /// Fused fallible path: a *valid* out-of-range value must make the cast fail, even when an
    /// invalid lane also holds an out-of-range value.
    #[test]
    fn cast_masked_valid_out_of_range_errors() {
        let arr = PrimitiveArray::new(
            buffer![1000u32, 300u32, 42u32],
            Validity::from_iter([false, true, true]),
        );
        #[expect(deprecated)]
        let err = arr
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::Nullable))
            .and_then(|a| a.to_canonical().map(|c| c.into_array()))
            .unwrap_err();
        assert!(matches!(err, VortexError::Compute(..)));
        assert!(err.to_string().contains("exceeds target range"));
    }

    /// Fused fallible path with no nulls: an out-of-range value must fail.
    #[test]
    fn cast_all_valid_out_of_range_errors() {
        let arr = buffer![10u32, 1000u32, 42u32].into_array();
        #[expect(deprecated)]
        let err = arr
            .cast(PType::U8.into())
            .and_then(|a| a.to_canonical().map(|c| c.into_array()))
            .unwrap_err();
        assert!(matches!(err, VortexError::Compute(..)));
    }

    /// Float-to-int through the fused path: out-of-range and non-finite *valid* values fail,
    /// while the same garbage at invalid positions is harmless.
    #[test]
    fn cast_f64_to_i32_masked() -> vortex_error::VortexResult<()> {
        // The invalid lane holds a value far outside i32 range; it must not trigger an error.
        let arr = PrimitiveArray::new(
            buffer![1e18f64, -5.0, 7.0],
            Validity::from_iter([false, true, true]),
        );
        #[expect(deprecated)]
        let casted = arr
            .into_array()
            .cast(DType::Primitive(PType::I32, Nullability::Nullable))?
            .to_primitive();
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_option_iter([None, Some(-5i32), Some(7)])
        );

        // A valid out-of-range float must fail.
        let arr = PrimitiveArray::new(
            buffer![1.0f64, 1e18, 7.0],
            Validity::from_iter([true, true, true]),
        );
        #[expect(deprecated)]
        let err = arr
            .into_array()
            .cast(DType::Primitive(PType::I32, Nullability::Nullable))
            .and_then(|a| a.to_canonical().map(|c| c.into_array()))
            .unwrap_err();
        assert!(matches!(err, VortexError::Compute(..)));
        Ok(())
    }

    #[rstest]
    #[case(buffer![0u8, 1, 2, 3, 255].into_array())]
    #[case(buffer![0u16, 100, 1000, 65535].into_array())]
    #[case(buffer![0u32, 100, 1000, 1000000].into_array())]
    #[case(buffer![0u64, 100, 1000, 1000000000].into_array())]
    #[case(buffer![-128i8, -1, 0, 1, 127].into_array())]
    #[case(buffer![-1000i16, -1, 0, 1, 1000].into_array())]
    #[case(buffer![-1000000i32, -1, 0, 1, 1000000].into_array())]
    #[case(buffer![-1000000000i64, -1, 0, 1, 1000000000].into_array())]
    #[case(buffer![0.0f32, 1.5, -2.5, 100.0, 1e6].into_array())]
    #[case(buffer![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0f32].into_array())]
    #[case(buffer![0.0f64, 1.5, -2.5, 100.0, 1e12].into_array())]
    #[case(buffer![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0f64].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1u8), None, Some(255), Some(0), None]).into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1i32), None, Some(-100), Some(0), None]).into_array())]
    #[case(buffer![42u32].into_array())]
    fn test_cast_primitive_conformance(#[case] array: crate::ArrayRef) {
        test_cast_conformance(&array);
    }
}

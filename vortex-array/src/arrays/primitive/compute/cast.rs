// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::NumCast;
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

/// Cast values from `F` to `T`. For infallible casts this is a pure pass. For fallible casts
/// where cached stats can't prove fit, the hot loop is unconditional `as_()` + a parallel range
/// check whose results OR-reduce into a single `fail_acc` word — one pass, no `?` in the inner
/// body, fully SIMD-vectorizable. If `fail_acc` is set, a cold scalar pass walks the array to
/// attribute the failure to a specific index for a precise error message.
fn cast_values<F, T>(
    array: ArrayView<'_, Primitive>,
    new_validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    let values = array.as_slice::<F>();

    if values_always_fit(F::PTYPE, T::PTYPE) || values_fit_in(array, T::PTYPE, ctx, false) {
        return Ok(PrimitiveArray::new(cast::<F, T>(values), new_validity).into_array());
    }

    let mask = array.validity()?.execute_mask(array.len(), ctx)?;
    let mut buffer = BufferMut::<T>::zeroed(values.len());
    let out = buffer.as_mut_slice();
    let mut fail_acc: u32 = 0;

    match &mask {
        Mask::AllFalse(_) => {
            // No valid lanes — buffer is already zeroed.
        }
        Mask::AllTrue(_) => {
            for (i, &v) in values.iter().enumerate() {
                out[i] = v.as_();
                fail_acc |= <T as NumCast>::from(v).is_none() as u32;
            }
        }
        Mask::Values(m) => {
            fail_acc = fallible_cast_with_validity::<F, T>(values, m.bit_buffer(), out);
        }
    }

    if fail_acc != 0 {
        // Cold scalar fallback: identify the failing index for a precise error.
        for (idx, (&v, valid)) in values.iter().zip(mask_iter(&mask, values.len())).enumerate() {
            if valid && <T as NumCast>::from(v).is_none() {
                vortex_bail!(
                    Compute: "Cannot cast {} to {} — value at index {} exceeds target range",
                    F::PTYPE, T::PTYPE, idx,
                );
            }
        }
        // Should be unreachable, but emit a generic error if the hot/cold paths disagree.
        vortex_bail!(
            Compute: "Cannot cast {} to {} — value exceeds target range",
            F::PTYPE, T::PTYPE,
        );
    }

    Ok(PrimitiveArray::new(buffer.freeze(), new_validity).into_array())
}

/// Unconditional `as_()` cast of every lane in `values` into `out`, with a SIMD-reducible
/// overflow detector that returns a nonzero failure word iff any valid lane would overflow `T`.
/// Walks validity in 64-lane blocks (`from_fn` lane-mask + uniform inner body, fully unrollable)
/// and bails at the block boundary on the first failure — branch is outside the SIMD region.
#[inline]
fn fallible_cast_with_validity<F, T>(
    values: &[F],
    bit_buffer: &BitBuffer,
    out: &mut [T],
) -> u32
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    debug_assert_eq!(values.len(), bit_buffer.len());
    debug_assert_eq!(values.len(), out.len());
    let bit_chunks = bit_buffer.chunks();
    let mut fail_acc: u32 = 0;
    let mut idx = 0usize;
    for word in bit_chunks.iter() {
        let valid: [bool; 64] = std::array::from_fn(|i| (word >> i) & 1 != 0);
        for i in 0..64 {
            let v = values[idx + i];
            out[idx + i] = v.as_();
            // Mask invalid lanes to F::zero (always fits any T) so they don't pollute fail_acc.
            let v_for_check = if valid[i] { v } else { F::zero() };
            fail_acc |= <T as NumCast>::from(v_for_check).is_none() as u32;
        }
        idx += 64;
        if fail_acc != 0 {
            return fail_acc;
        }
    }
    let rem = bit_chunks.remainder_bits();
    for b in 0..bit_chunks.remainder_len() {
        let v = values[idx + b];
        out[idx + b] = v.as_();
        let valid = (rem >> b) & 1 != 0;
        let v_for_check = if valid { v } else { F::zero() };
        fail_acc |= <T as NumCast>::from(v_for_check).is_none() as u32;
    }
    fail_acc
}

/// Cold-path iterator over a `Mask` as a sequence of `bool`s. Only used after `fail_acc != 0`
/// to attribute the failure to a specific index.
fn mask_iter<'a>(mask: &'a Mask, len: usize) -> Box<dyn Iterator<Item = bool> + 'a> {
    match mask {
        Mask::AllTrue(_) => Box::new(std::iter::repeat_n(true, len)),
        Mask::AllFalse(_) => Box::new(std::iter::repeat_n(false, len)),
        Mask::Values(m) => Box::new(m.bit_buffer().iter()),
    }
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

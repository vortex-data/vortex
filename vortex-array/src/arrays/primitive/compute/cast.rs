// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
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
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::match_each_native_ptype;
use crate::scalar::PValue;
use crate::scalar::ScalarValue;
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

        // Same ptype: zero-copy, just update validity.
        if src_ptype == new_ptype {
            return Ok(Some(reinterpret(array, src_ptype, new_validity)));
        }

        let same_width = src_ptype.byte_width() == new_ptype.byte_width();
        let same_int_rep = same_width && src_ptype.is_int() && new_ptype.is_int();
        let infallible = values_always_fit(src_ptype, new_ptype);

        // Same-width int↔int: bit patterns are identical under 2's complement, so once we have
        // verified the values fit we can just reinterpret the buffer with no per-element work.
        if same_int_rep {
            if !infallible && !values_fit_in(array, new_ptype, ctx) {
                vortex_bail!(
                    Compute: "Cannot cast {} to {} — values exceed target range",
                    src_ptype, new_ptype,
                );
            }
            return Ok(Some(reinterpret(array, new_ptype, new_validity)));
        }

        // Infallible casts (e.g. int → wider int, int → float): no min/max needed. Same-width
        // casts try to reuse the source allocation when uniquely owned; otherwise allocate.
        if infallible {
            return Ok(Some(match_each_native_ptype!(new_ptype, |T| {
                match_each_native_ptype!(src_ptype, |F| {
                    PrimitiveArray::new(cast_or_reuse::<F, T>(array), new_validity).into_array()
                })
            })));
        }

        // Fallible cast (e.g. float → int, narrowing int, narrowing float): cast and min/max
        // validation happen in a single pass over the source values.
        Ok(Some(match_each_native_ptype!(new_ptype, |T| {
            match_each_native_ptype!(src_ptype, |F| {
                cast_with_check::<F, T>(array, new_validity, ctx)?
            })
        })))
    }
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

/// Whether every valid value of `src` is guaranteed representable in `target` without inspecting
/// the values themselves.
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

/// Returns `true` if all valid values in `array` are representable as `target_ptype`. Uses cached
/// min/max statistics when available and otherwise computes them with a single pass.
fn values_fit_in(
    array: ArrayView<'_, Primitive>,
    target_ptype: PType,
    ctx: &mut ExecutionCtx,
) -> bool {
    let target_dtype = DType::Primitive(target_ptype, Nullability::NonNullable);
    aggregate_fn::fns::min_max::min_max(array.array(), ctx)
        .ok()
        .flatten()
        .is_none_or(|mm| mm.min.cast(&target_dtype).is_ok() && mm.max.cast(&target_dtype).is_ok())
}

/// Cast values from `F` to `T`, reusing the source buffer in-place when the widths match and the
/// underlying allocation is uniquely owned. Falls back to a fresh allocation otherwise.
fn cast_or_reuse<F, T>(array: ArrayView<'_, Primitive>) -> Buffer<T>
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    if size_of::<F>() == size_of::<T>() {
        let source = Buffer::<F>::from_byte_buffer(array.buffer_handle().to_host_sync());
        source
            .map_each_in_place(<F as AsPrimitive<T>>::as_)
            .freeze()
    } else {
        cast::<F, T>(array.as_slice())
    }
}

/// Fallible cast: writes target values to a new buffer and tracks the source min/max in the same
/// pass, then bails if the min/max indicate values overflow the target type.
fn cast_with_check<F, T>(
    array: ArrayView<'_, Primitive>,
    new_validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType,
    PValue: From<F>,
{
    let target_dtype = DType::Primitive(T::PTYPE, Nullability::NonNullable);

    // If min/max stats are already cached we can decide the fit without revisiting the values.
    if let Some(fits) = cached_min_max_fits(array, &target_dtype) {
        if !fits {
            vortex_bail!(
                Compute: "Cannot cast {} to {} — values exceed target range",
                F::PTYPE, T::PTYPE,
            );
        }
        return Ok(PrimitiveArray::new(cast_or_reuse::<F, T>(array), new_validity).into_array());
    }

    let values = array.as_slice::<F>();
    let mask = array.validity()?.execute_mask(array.len(), ctx)?;
    let mut min_max: Option<(F, F)> = None;

    let buffer: Buffer<T> = match &mask {
        Mask::AllTrue(_) => BufferMut::from_trusted_len_iter(values.iter().map(|&v| {
            track_min_max(&mut min_max, v);
            v.as_()
        }))
        .freeze(),
        Mask::AllFalse(_) => cast::<F, T>(values),
        Mask::Values(m) => BufferMut::from_trusted_len_iter(
            values.iter().zip(m.bit_buffer().iter()).map(|(&v, valid)| {
                if valid {
                    track_min_max(&mut min_max, v);
                }
                v.as_()
            }),
        )
        .freeze(),
    };

    if let Some((min, max)) = min_max {
        let stats = array.array().statistics();
        stats.set(
            Stat::Min,
            Precision::Exact(ScalarValue::Primitive(PValue::from(min))),
        );
        stats.set(
            Stat::Max,
            Precision::Exact(ScalarValue::Primitive(PValue::from(max))),
        );

        if PValue::from(min).cast::<T>().is_err() || PValue::from(max).cast::<T>().is_err() {
            vortex_bail!(
                Compute: "Cannot cast {} to {} — values exceed target range",
                F::PTYPE, T::PTYPE,
            );
        }
    }

    Ok(PrimitiveArray::new(buffer, new_validity).into_array())
}

fn track_min_max<F: NativePType>(min_max: &mut Option<(F, F)>, value: F) {
    if value.is_nan() {
        return;
    }
    match min_max {
        None => *min_max = Some((value, value)),
        Some((min, max)) => {
            if value.total_compare(*min).is_lt() {
                *min = value;
            } else if value.total_compare(*max).is_gt() {
                *max = value;
            }
        }
    }
}

/// Returns `Some(true)` if cached min/max prove every valid value fits in `target_dtype`,
/// `Some(false)` if cached min/max prove the cast must fail, and `None` if stats are unavailable.
fn cached_min_max_fits(array: ArrayView<'_, Primitive>, target_dtype: &DType) -> Option<bool> {
    let stats = array.array().statistics();
    let min = stats.get(Stat::Min).and_then(Precision::as_exact)?;
    let max = stats.get(Stat::Max).and_then(Precision::as_exact)?;
    Some(min.cast(target_dtype).is_ok() && max.cast(target_dtype).is_ok())
}

/// Out-of-range values at invalid positions are truncated/wrapped by `as`, which is fine because
/// they are masked out by validity.
fn cast<F: NativePType + AsPrimitive<T>, T: NativePType>(array: &[F]) -> Buffer<T> {
    BufferMut::from_trusted_len_iter(array.iter().map(|&src| src.as_())).freeze()
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

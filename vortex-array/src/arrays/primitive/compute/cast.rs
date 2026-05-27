// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::NumCast;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::lane_ops_indexed::ReinterpretSink;
use vortex_buffer::lane_ops_indexed::map_no_validity;
use vortex_buffer::lane_ops_indexed::map_no_validity_in_place;
use vortex_buffer::lane_ops_indexed::try_map_no_validity;
use vortex_buffer::lane_ops_indexed::try_map_no_validity_in_place;
use vortex_buffer::lane_ops_indexed::try_map_with_mask;
use vortex_buffer::lane_ops_indexed::try_map_with_mask_in_place;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
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

/// Cast values from `F` to `T`. Always routes through the fallible lane-op kernels with
/// `NumCast::from`. The kernel branches once on the mask shape:
///
/// - `Mask::AllTrue`  → [`try_map_no_validity`] — no per-lane validity work.
/// - `Mask::AllFalse` → bulk zero — the closure is never invoked.
/// - `Mask::Values`   → [`try_map_with_mask`] — the closure neutralizes null lanes
///   via the `* valid as F` multiply trick so out-of-range null-lane values don't
///   trigger spurious errors.
///
/// For statically-infallible casts (e.g. widening) LLVM proves `NumCast::from` always
/// returns `Some` and strips the fail-tracking machinery, generating the same bare
/// `ushll` widen loop the old hand-written `as_()` fast path produced.
fn cast_values<F, T>(
    array: ArrayView<'_, Primitive>,
    new_validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    let overflow = || {
        vortex_err!(
            Compute: "Cannot cast {} to {} — value exceeds target range",
            F::PTYPE, T::PTYPE,
        )
    };

    // Returns `true` if every value of `from` is representable in `to` without loss.
    //
    // Equivalent to `from.least_supertype(to) == Some(to)`, i.e. the value domain of `from`
    // is a subset of `to`'s. This is the static-only check — it does not consult any array
    // statistics. Used to short-circuit checked casts when the conversion is infallible by
    // type alone (widening uint→uint, signed→signed, u8→i16, i32→f64, etc.).
    fn casts_losslessly_to(from: PType, to: PType) -> bool {
        from.least_supertype(to) == Some(to)
    }

    // Skip the fallible kernel when the conversion is infallible by type alone (widening) or
    // when cached min/max prove every value fits in `T`.
    let target_dtype = DType::Primitive(T::PTYPE, Nullability::NonNullable);
    let infallible = casts_losslessly_to(F::PTYPE, T::PTYPE)
        || cached_values_fit_in(array, &target_dtype) == Some(true);

    let len = array.len();

    // Same-bit-width in-place fast path: when F and T have the same byte width, try to take
    // unique ownership of the buffer. If successful, each kernel call site below mutates in
    // place via `ReinterpretSink` and transmutes the wrapper at the end, saving the output
    // allocation. Falls back to the out-of-place path (borrowed slice + fresh buffer) when
    // the buffer is shared — the common case under the current borrow-based kernel API.
    let same_bit_width = F::PTYPE.byte_width() == T::PTYPE.byte_width();
    let owned: Option<BufferMut<F>> = if same_bit_width {
        array.into_owned().try_into_buffer_mut::<F>().ok()
    } else {
        None
    };
    let values: &[F] = array.as_slice::<F>();

    if infallible {
        // Truncating `as`-cast — safe here because static type analysis or cached stats prove
        // every valid value fits. Null lanes' underlying garbage gets truncated/wrapped
        // (harmless: the result validity bitmap masks them downstream).
        return match owned {
            Some(mut buf) => {
                map_no_validity_in_place(
                    ReinterpretSink::<F, T>::new(buf.as_mut_slice()),
                    |v: F| v.as_(),
                );
                // SAFETY: same size + alignment for NativePType same-byte-width pairs;
                // every F-slot was overwritten with a real `T` bit pattern.
                let result: BufferMut<T> = unsafe { buf.transmute::<T>() };
                Ok(PrimitiveArray::new(result.freeze(), new_validity).into_array())
            }
            None => {
                let mut buffer = BufferMut::<T>::with_capacity(len);
                map_no_validity(values, &mut buffer.spare_capacity_mut()[..len], |v| v.as_());
                // SAFETY: map_no_validity initializes every lane.
                unsafe { buffer.set_len(len) };
                Ok(PrimitiveArray::new(buffer.freeze(), new_validity).into_array())
            }
        };
    }

    let mask = array.validity()?.execute_mask(len, ctx)?;

    let buffer: Buffer<T> = match (&mask, owned) {
        (Mask::AllTrue(_), Some(mut buf)) => {
            try_map_no_validity_in_place(
                ReinterpretSink::<F, T>::new(buf.as_mut_slice()),
                |v: F| <T as NumCast>::from(v),
            )
            .map_err(|_| overflow())?;
            // SAFETY: same size + alignment for NativePType same-byte-width pairs;
            // every F-slot now holds a `T` bit pattern written by `ReinterpretSink`.
            let result: BufferMut<T> = unsafe { buf.transmute::<T>() };
            result.freeze()
        }
        (Mask::AllTrue(_), None) => {
            let mut buffer = BufferMut::<T>::with_capacity(len);
            try_map_no_validity(values, &mut buffer.spare_capacity_mut()[..len], |v| {
                <T as NumCast>::from(v)
            })
            .map_err(|_| overflow())?;
            // SAFETY: try_map_no_validity returned Ok, so it initialized every lane.
            unsafe { buffer.set_len(len) };
            buffer.freeze()
        }
        (Mask::AllFalse(_), Some(buf)) => {
            // SAFETY: same size + alignment by NativePType same-byte-width invariant.
            let mut t_buf: BufferMut<T> = unsafe { buf.transmute::<T>() };
            t_buf.as_mut_slice().fill(T::zero());
            t_buf.freeze()
        }
        (Mask::AllFalse(_), None) => BufferMut::<T>::zeroed(len).freeze(),
        (Mask::Values(m), Some(mut buf)) => {
            try_map_with_mask_in_place(
                ReinterpretSink::<F, T>::new(buf.as_mut_slice()),
                m.bit_buffer(),
                |v: F, valid| <T as NumCast>::from(v).or_else(|| (!valid).then(T::zero)),
            )
            .map_err(|_| overflow())?;
            // SAFETY: same size + alignment for NativePType same-byte-width pairs;
            // every F-slot now holds a `T` bit pattern written by `ReinterpretSink`.
            let result: BufferMut<T> = unsafe { buf.transmute::<T>() };
            result.freeze()
        }
        (Mask::Values(m), None) => {
            let mut buffer = BufferMut::<T>::with_capacity(len);
            try_map_with_mask(
                values,
                m.bit_buffer(),
                &mut buffer.spare_capacity_mut()[..len],
                // Lazy validity: only consult `valid` on the failure branch. For widening /
                // statically-infallible casts, `NumCast::from` is always `Some` so the
                // `or_else` is provably dead — LLVM DCEs the validity path entirely, giving
                // the same codegen as the maskless kernel. For narrowing, `valid` is only
                // read at lanes that actually overflowed (a cold check on top of the cast).
                |v, valid| <T as NumCast>::from(v).or_else(|| (!valid).then(T::zero)),
            )
            .map_err(|_| overflow())?;
            // SAFETY: try_map_with_mask returned Ok, so it initialized every lane.
            unsafe { buffer.set_len(len) };
            buffer.freeze()
        }
    };

    Ok(PrimitiveArray::new(buffer, new_validity).into_array())
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

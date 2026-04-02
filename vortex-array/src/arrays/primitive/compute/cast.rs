// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn;
use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::cast::CastKernel;

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

        // First, check that the cast is compatible with the source array's validity
        let new_validity = array
            .validity()
            .cast_nullability(new_nullability, array.len())?;

        // Same ptype: zero-copy, just update validity.
        if array.ptype() == new_ptype {
            // SAFETY: validity and data buffer still have same length
            return Ok(Some(unsafe {
                PrimitiveArray::new_unchecked_from_handle(
                    array.buffer_handle().clone(),
                    array.ptype(),
                    new_validity,
                )
                .into_array()
            }));
        }

        // Same-width integers have identical bit representations due to 2's
        // complement. If all values fit in the target range, reinterpret with
        // no allocation.
        if array.ptype().is_int()
            && new_ptype.is_int()
            && array.ptype().byte_width() == new_ptype.byte_width()
        {
            if !values_fit_in(array, new_ptype, ctx) {
                vortex_bail!(
                    Compute: "Cannot cast {} to {} — values exceed target range",
                    array.ptype(),
                    new_ptype,
                );
            }
            // SAFETY: both types are integers with the same size and alignment, and
            // min/max confirm all valid values are representable in the target type.
            return Ok(Some(unsafe {
                PrimitiveArray::new_unchecked_from_handle(
                    array.buffer_handle().clone(),
                    new_ptype,
                    new_validity,
                )
                .into_array()
            }));
        }

        let mask = array.validity_mask();

        // Otherwise, we need to cast the values one-by-one.
        Ok(Some(match_each_native_ptype!(new_ptype, |T| {
            match_each_native_ptype!(array.ptype(), |F| {
                PrimitiveArray::new(cast::<F, T>(array.as_slice(), mask)?, new_validity)
                    .into_array()
            })
        })))
    }
}

/// Returns `true` if all valid values in `array` are representable as `target_ptype`.
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

fn cast<F: NativePType, T: NativePType>(array: &[F], mask: Mask) -> VortexResult<Buffer<T>> {
    let try_cast = |src: F| -> VortexResult<T> {
        T::from(src).ok_or_else(|| vortex_err!(Compute: "Failed to cast {} to {:?}", src, T::PTYPE))
    };
    match mask.bit_buffer() {
        AllOr::None => Ok(Buffer::zeroed(array.len())),
        AllOr::All => {
            let mut buffer = BufferMut::with_capacity(array.len());
            for &src in array {
                // SAFETY: we've pre-allocated the required capacity
                unsafe { buffer.push_unchecked(try_cast(src)?) }
            }
            Ok(buffer.freeze())
        }
        AllOr::Some(b) => {
            let mut buffer = BufferMut::with_capacity(array.len());
            for (&src, valid) in array.iter().zip(b.iter()) {
                let dst = if valid { try_cast(src)? } else { T::default() };
                // SAFETY: we've pre-allocated the required capacity
                unsafe { buffer.push_unchecked(dst) }
            }
            Ok(buffer.freeze())
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexError;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::canonical::ToCanonical;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    #[allow(clippy::cognitive_complexity)]
    #[test]
    fn cast_u32_u8() {
        let arr = buffer![0u32, 10, 200].into_array();

        // cast from u32 to u8
        let p = arr.cast(PType::U8.into()).unwrap().to_primitive();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([0u8, 10, 200]));
        assert!(matches!(p.validity(), Validity::NonNullable));

        // to nullable
        let p = p
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::Nullable))
            .unwrap()
            .to_primitive();
        assert_arrays_eq!(
            p,
            PrimitiveArray::new(buffer![0u8, 10, 200], Validity::AllValid)
        );
        assert!(matches!(p.validity(), Validity::AllValid));

        // back to non-nullable
        let p = p
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::NonNullable))
            .unwrap()
            .to_primitive();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([0u8, 10, 200]));
        assert!(matches!(p.validity(), Validity::NonNullable));

        // to nullable u32
        let p = p
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap()
            .to_primitive();
        assert_arrays_eq!(
            p,
            PrimitiveArray::new(buffer![0u32, 10, 200], Validity::AllValid)
        );
        assert!(matches!(p.validity(), Validity::AllValid));

        // to non-nullable u8
        let p = p
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::NonNullable))
            .unwrap()
            .to_primitive();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([0u8, 10, 200]));
        assert!(matches!(p.validity(), Validity::NonNullable));
    }

    #[test]
    fn cast_u32_f32() {
        let arr = buffer![0u32, 10, 200].into_array();
        let u8arr = arr.cast(PType::F32.into()).unwrap().to_primitive();
        assert_arrays_eq!(u8arr, PrimitiveArray::from_iter([0.0f32, 10., 200.]));
    }

    #[test]
    fn cast_i32_u32() {
        let arr = buffer![-1i32].into_array();
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
            p.validity_mask().unwrap(),
            Mask::from(BitBuffer::from(vec![false, true, true]))
        );
    }

    /// Same-width integer cast where all values fit: should reinterpret the
    /// buffer without allocation (pointer identity).
    #[test]
    fn cast_same_width_int_reinterprets_buffer() -> vortex_error::VortexResult<()> {
        let src = PrimitiveArray::from_iter([0u32, 10, 100]);
        let src_ptr = src.as_slice::<u32>().as_ptr();

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
        let casted = arr
            .into_array()
            .cast(DType::Primitive(PType::I8, Nullability::Nullable))?
            .to_primitive();
        assert_eq!(casted.len(), 2);
        assert!(matches!(casted.validity(), Validity::AllInvalid));
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
    #[case(buffer![0.0f64, 1.5, -2.5, 100.0, 1e12].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1u8), None, Some(255), Some(0), None]).into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1i32), None, Some(-100), Some(0), None]).into_array())]
    #[case(buffer![42u32].into_array())]
    fn test_cast_primitive_conformance(#[case] array: crate::ArrayRef) {
        test_cast_conformance(&array);
    }
}

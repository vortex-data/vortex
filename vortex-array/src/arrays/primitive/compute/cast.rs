// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::vtable::ValidityHelper;

impl CastKernel for Primitive {
    fn cast(
        array: &PrimitiveArray,
        dtype: &DType,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let DType::Primitive(new_ptype, new_nullability) = dtype else {
            return Ok(None);
        };
        let (new_ptype, new_nullability) = (*new_ptype, *new_nullability);

        // First, check that the cast is compatible with the source array's validity
        let new_validity = array
            .validity()
            .clone()
            .cast_nullability(new_nullability, array.len())?;

        // If the bit width is the same, we can short-circuit and simply update the validity
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

        let mask = array.validity_mask()?;

        // Otherwise, we need to cast the values one-by-one
        Ok(Some(match_each_native_ptype!(new_ptype, |T| {
            match_each_native_ptype!(array.ptype(), |F| {
                PrimitiveArray::new(cast::<F, T>(array.as_slice(), mask)?, new_validity)
                    .into_array()
            })
        })))
    }
}

fn cast<F: NativePType, T: NativePType>(array: &[F], mask: Mask) -> VortexResult<Buffer<T>> {
    match mask.bit_buffer() {
        AllOr::All => {
            let mut buffer = BufferMut::with_capacity(array.len());
            for item in array {
                let item = T::from(*item).ok_or_else(
                    || vortex_err!(Compute: "Failed to cast {} to {:?}", item, T::PTYPE),
                )?;
                // SAFETY: we've pre-allocated the required capacity
                unsafe { buffer.push_unchecked(item) }
            }
            Ok(buffer.freeze())
        }
        AllOr::None => Ok(Buffer::zeroed(array.len())),
        AllOr::Some(b) => {
            // TODO(robert): Depending on density of the buffer might be better to prefill Buffer and only write valid values
            let mut buffer = BufferMut::with_capacity(array.len());
            for (item, valid) in array.iter().zip(b.iter()) {
                if valid {
                    let item = T::from(*item).ok_or_else(
                        || vortex_err!(Compute: "Failed to cast {} to {:?}", item, T::PTYPE),
                    )?;
                    // SAFETY: we've pre-allocated the required capacity
                    unsafe { buffer.push_unchecked(item) }
                } else {
                    // SAFETY: we've pre-allocated the required capacity
                    unsafe { buffer.push_unchecked(T::default()) }
                }
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
    use crate::vtable::ValidityHelper;

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
        assert!(error.to_string().contains("Failed to cast -1 to U32"));
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

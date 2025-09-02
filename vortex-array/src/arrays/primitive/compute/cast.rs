// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{DType, NativePType, Nullability, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail, vortex_err};

use crate::arrays::PrimitiveVTable;
use crate::arrays::primitive::PrimitiveArray;
use crate::compute::{CastKernel, CastKernelAdapter};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl CastKernel for PrimitiveVTable {
    fn cast(&self, array: &PrimitiveArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let DType::Primitive(new_ptype, new_nullability) = dtype else {
            return Ok(None);
        };
        let (new_ptype, new_nullability) = (*new_ptype, *new_nullability);

        // First, check that the cast is compatible with the source array's validity
        let new_validity = if array.dtype().nullability() == new_nullability {
            array.validity().clone()
        } else if new_nullability == Nullability::Nullable {
            // from non-nullable to nullable
            array.validity().clone().into_nullable()
        } else if new_nullability == Nullability::NonNullable && array.validity().all_valid() {
            // from nullable but all valid, to non-nullable
            Validity::NonNullable
        } else {
            vortex_bail!(
                "invalid cast from nullable to non-nullable, since source array actually contains nulls"
            );
        };

        // If the bit width is the same, we can short-circuit and simply update the validity
        if array.ptype() == new_ptype {
            return Ok(Some(
                PrimitiveArray::from_byte_buffer(
                    array.byte_buffer().clone(),
                    array.ptype(),
                    new_validity,
                )
                .into_array(),
            ));
        }

        // Otherwise, we need to cast the values one-by-one
        match_each_native_ptype!(new_ptype, |T| {
            Ok(Some(
                PrimitiveArray::new(cast::<T>(array)?, new_validity).into_array(),
            ))
        })
    }
}

register_kernel!(CastKernelAdapter(PrimitiveVTable).lift());

fn cast<T: NativePType>(array: &PrimitiveArray) -> VortexResult<Buffer<T>> {
    let mut buffer = BufferMut::with_capacity(array.len());
    match_each_native_ptype!(array.ptype(), |P| {
        for item in array.as_slice::<P>() {
            let item = T::from(*item).ok_or_else(
                || vortex_err!(ComputeError: "Failed to cast {} to {:?}", item, T::PTYPE),
            )?;
            // SAFETY: we've pre-allocated the required capacity
            unsafe { buffer.push_unchecked(item) }
        }
    });
    Ok(buffer.freeze())
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_error::VortexError;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::canonical::ToCanonical;
    use crate::compute::cast;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;

    #[test]
    fn cast_u32_u8() {
        let arr = buffer![0u32, 10, 200].into_array();

        // cast from u32 to u8
        let p = cast(&arr, PType::U8.into()).unwrap().to_primitive();
        assert_eq!(p.as_slice::<u8>(), vec![0u8, 10, 200]);
        assert_eq!(p.validity(), &Validity::NonNullable);

        // to nullable
        let p = cast(
            p.as_ref(),
            &DType::Primitive(PType::U8, Nullability::Nullable),
        )
        .unwrap()
        .to_primitive();
        assert_eq!(p.as_slice::<u8>(), vec![0u8, 10, 200]);
        assert_eq!(p.validity(), &Validity::AllValid);

        // back to non-nullable
        let p = cast(
            p.as_ref(),
            &DType::Primitive(PType::U8, Nullability::NonNullable),
        )
        .unwrap()
        .to_primitive();
        assert_eq!(p.as_slice::<u8>(), vec![0u8, 10, 200]);
        assert_eq!(p.validity(), &Validity::NonNullable);

        // to nullable u32
        let p = cast(
            p.as_ref(),
            &DType::Primitive(PType::U32, Nullability::Nullable),
        )
        .unwrap()
        .to_primitive();
        assert_eq!(p.as_slice::<u32>(), vec![0u32, 10, 200]);
        assert_eq!(p.validity(), &Validity::AllValid);

        // to non-nullable u8
        let p = cast(
            p.as_ref(),
            &DType::Primitive(PType::U8, Nullability::NonNullable),
        )
        .unwrap()
        .to_primitive();
        assert_eq!(p.as_slice::<u8>(), vec![0u8, 10, 200]);
        assert_eq!(p.validity(), &Validity::NonNullable);
    }

    #[test]
    fn cast_u32_f32() {
        let arr = buffer![0u32, 10, 200].into_array();
        let u8arr = cast(&arr, PType::F32.into()).unwrap().to_primitive();
        assert_eq!(u8arr.as_slice::<f32>(), vec![0.0f32, 10., 200.]);
    }

    #[test]
    fn cast_i32_u32() {
        let arr = buffer![-1i32].into_array();
        let error = cast(&arr, PType::U32.into()).err().unwrap();
        let VortexError::ComputeError(s, _) = error else {
            unreachable!()
        };
        assert_eq!(s.to_string(), "Failed to cast -1 to U32");
    }

    #[test]
    fn cast_array_with_nulls_to_nonnullable() {
        let arr = PrimitiveArray::from_option_iter([Some(-1i32), None, Some(10)]);
        let err = cast(arr.as_ref(), PType::I32.into()).unwrap_err();
        let VortexError::InvalidArgument(s, _) = err else {
            unreachable!()
        };
        assert_eq!(
            s.to_string(),
            "invalid cast from nullable to non-nullable, since source array actually contains nulls"
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
        test_cast_conformance(array.as_ref());
    }
}

use vortex_dtype::{match_each_native_ptype, DType, NativePType, Nullability};
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::unary::CastFn;
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayData, ArrayLen, IntoArrayData};

impl CastFn<PrimitiveArray> for PrimitiveEncoding {
    fn cast(&self, array: &PrimitiveArray, dtype: &DType) -> VortexResult<ArrayData> {
        let DType::Primitive(new_ptype, new_nullability) = dtype else {
            vortex_bail!(MismatchedTypes: "primitive type", dtype);
        };
        let (new_ptype, new_nullability) = (*new_ptype, *new_nullability);

        // First, check that the cast is compatible with the source array's validity
        let new_validity = if array.dtype().nullability() == new_nullability {
            array.validity()
        } else if new_nullability == Nullability::Nullable {
            // from non-nullable to nullable
            array.validity().into_nullable()
        } else if new_nullability == Nullability::NonNullable
            && array.validity().to_logical(array.len()).all_valid()
        {
            // from nullable but all valid, to non-nullable
            Validity::NonNullable
        } else {
            vortex_bail!("invalid cast from nullable to non-nullable, since source array actually contains nulls");
        };

        // If the bit width is the same, we can short-circuit and simply update the validity
        if array.ptype() == new_ptype {
            return Ok(
                PrimitiveArray::new(array.buffer().clone(), array.ptype(), new_validity)
                    .into_array(),
            );
        }

        // Otherwise, we need to cast the values one-by-one
        match_each_native_ptype!(new_ptype, |$T| {
            Ok(PrimitiveArray::from_vec(
                cast::<$T>(array)?,
                new_validity,
            ).into_array())
        })
    }
}

fn cast<T: NativePType>(array: &PrimitiveArray) -> VortexResult<Vec<T>> {
    match_each_native_ptype!(array.ptype(), |$E| {
        array
            .maybe_null_slice::<$E>()
            .iter()
            // TODO(ngates): allow configurable checked/unchecked casting
            .map(|&v| {
                T::from(v).ok_or_else(|| {
                    vortex_err!(ComputeError: "Failed to cast {} to {:?}", v, T::PTYPE)
                })
            })
            .collect()
    })
}

#[cfg(test)]
mod test {
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_error::VortexError;

    use crate::array::PrimitiveArray;
    use crate::compute::unary::try_cast;
    use crate::validity::Validity;
    use crate::{IntoArrayData, IntoArrayVariant};

    #[test]
    fn cast_u32_u8() {
        let arr = vec![0u32, 10, 200].into_array();

        // cast from u32 to u8
        let p = try_cast(&arr, PType::U8.into())
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(p.maybe_null_slice::<u8>(), vec![0u8, 10, 200]);
        assert_eq!(p.validity(), Validity::NonNullable);

        // to nullable
        let p = try_cast(&p, &DType::Primitive(PType::U8, Nullability::Nullable))
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(p.maybe_null_slice::<u8>(), vec![0u8, 10, 200]);
        assert_eq!(p.validity(), Validity::AllValid);

        // back to non-nullable
        let p = try_cast(&p, &DType::Primitive(PType::U8, Nullability::NonNullable))
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(p.maybe_null_slice::<u8>(), vec![0u8, 10, 200]);
        assert_eq!(p.validity(), Validity::NonNullable);

        // to nullable u32
        let p = try_cast(&p, &DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(p.maybe_null_slice::<u32>(), vec![0u32, 10, 200]);
        assert_eq!(p.validity(), Validity::AllValid);

        // to non-nullable u8
        let p = try_cast(&p, &DType::Primitive(PType::U8, Nullability::NonNullable))
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(p.maybe_null_slice::<u8>(), vec![0u8, 10, 200]);
        assert_eq!(p.validity(), Validity::NonNullable);
    }

    #[test]
    fn cast_u32_f32() {
        let arr = vec![0u32, 10, 200].into_array();
        let u8arr = try_cast(&arr, PType::F32.into())
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(u8arr.maybe_null_slice::<f32>(), vec![0.0f32, 10., 200.]);
    }

    #[test]
    fn cast_i32_u32() {
        let arr = vec![-1i32].into_array();
        let error = try_cast(&arr, PType::U32.into()).err().unwrap();
        let VortexError::ComputeError(s, _) = error else {
            unreachable!()
        };
        assert_eq!(s.to_string(), "Failed to cast -1 to U32");
    }

    #[test]
    fn cast_array_with_nulls_to_nonnullable() {
        let arr = PrimitiveArray::from_nullable_vec(vec![Some(-1i32), None, Some(10)]).into_array();
        let err = try_cast(&arr, PType::I32.into()).unwrap_err();
        let VortexError::InvalidArgument(s, _) = err else {
            unreachable!()
        };
        assert_eq!(s.to_string(), "invalid cast from nullable to non-nullable, since source array actually contains nulls");
    }
}

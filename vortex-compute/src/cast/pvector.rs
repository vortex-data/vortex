// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::NumCast;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_vector::Scalar;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::primitive::PScalar;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PrimitiveScalar;
use vortex_vector::primitive::PrimitiveVector;

use crate::cast::Cast;
use crate::cast::try_cast_scalar_common;
use crate::cast::try_cast_vector_common;

impl<T: NativePType> Cast for PVector<T> {
    type Output = Vector;

    /// Cast a primitive vector to a different primitive type.
    fn cast(&self, target_dtype: &DType) -> VortexResult<Vector> {
        if let Some(result) = try_cast_vector_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // We have the same `PType` and we have compatible nullability.
            DType::Primitive(target_ptype, n)
                if *target_ptype == T::PTYPE && (n.is_nullable() || self.validity().all_true()) =>
            {
                Ok(self.clone().into())
            }
            // We can possibly convert to the target `PType` and we have compatible nullability.
            DType::Primitive(target_ptype, n) if n.is_nullable() || self.validity().all_true() => {
                match_each_native_ptype!(*target_ptype, |Dst| {
                    let result = cast_pvector::<T, Dst>(self)?;
                    Ok(PrimitiveVector::from(result).into())
                })
            }
            _ => {
                vortex_bail!("Cannot cast PVector<{}> to {}", T::PTYPE, target_dtype);
            }
        }
    }
}

/// Cast a [`PVector<F>`] to a [`PVector<T>`] by converting each element.
///
/// Returns an error if any valid element cannot be converted (e.g., overflow).
fn cast_pvector<Src: NativePType, Dst: NativePType>(
    src: &PVector<Src>,
) -> VortexResult<PVector<Dst>> {
    let elements: &[Src] = src.as_ref();
    match src.validity().bit_buffer() {
        AllOr::All => {
            let mut buffer = BufferMut::with_capacity(elements.len());
            for &item in elements {
                let converted = <Dst as NumCast>::from(item).ok_or_else(
                    || vortex_err!(ComputeError: "Failed to cast {} to {:?}", item, Dst::PTYPE),
                )?;
                // SAFETY: We pre-allocated the required capacity.
                unsafe { buffer.push_unchecked(converted) }
            }
            Ok(PVector::from(buffer.freeze()))
        }
        AllOr::None => Ok(PVector::new(
            Buffer::zeroed(elements.len()),
            Mask::new_false(elements.len()),
        )),
        AllOr::Some(bit_buffer) => {
            let mut buffer = BufferMut::with_capacity(elements.len());
            for (&item, valid) in elements.iter().zip(bit_buffer.iter()) {
                if valid {
                    let converted = <Dst as NumCast>::from(item).ok_or_else(
                        || vortex_err!(ComputeError: "Failed to cast {} to {:?}", item, Dst::PTYPE),
                    )?;
                    // SAFETY: We pre-allocated the required capacity.
                    unsafe { buffer.push_unchecked(converted) }
                } else {
                    // SAFETY: We pre-allocated the required capacity.
                    unsafe { buffer.push_unchecked(Dst::default()) }
                }
            }
            Ok(PVector::new(buffer.freeze(), src.validity().clone()))
        }
    }
}

impl<T: NativePType> Cast for PScalar<T> {
    type Output = Scalar;

    /// Cast a primitive scalar to a different primitive type.
    fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if let Some(result) = try_cast_scalar_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // We have the same `PType` and we have compatible nullability.
            DType::Primitive(target_ptype, n)
                if *target_ptype == T::PTYPE && (n.is_nullable() || self.is_valid()) =>
            {
                Ok(self.clone().into())
            }
            // We can possibly convert to the target `PType` and we have compatible nullability.
            DType::Primitive(target_ptype, n) if n.is_nullable() || self.is_valid() => {
                match_each_native_ptype!(*target_ptype, |Dst| {
                    let result = match self.value() {
                        None => PScalar::null(),
                        Some(v) => {
                            let converted = <Dst as NumCast>::from(v).ok_or_else(|| {
                                vortex_err!(ComputeError: "Failed to cast {} to {:?}", v, Dst::PTYPE)
                            })?;
                            PScalar::new(Some(converted))
                        }
                    };
                    Ok(PrimitiveScalar::from(result).into())
                })
            }
            _ => {
                vortex_bail!("Cannot cast PScalar<{}> to {}", T::PTYPE, target_dtype);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_dtype::PTypeDowncast;
    use vortex_error::VortexError;
    use vortex_mask::Mask;
    use vortex_vector::ScalarOps;
    use vortex_vector::VectorOps;
    use vortex_vector::primitive::PScalar;
    use vortex_vector::primitive::PVector;

    use crate::cast::Cast;

    #[rstest]
    #[case(PType::U8)]
    #[case(PType::U16)]
    #[case(PType::U32)]
    #[case(PType::U64)]
    #[case(PType::I8)]
    #[case(PType::I16)]
    #[case(PType::I32)]
    #[case(PType::I64)]
    #[case(PType::F32)]
    #[case(PType::F64)]
    fn cast_u32_to_ptype(#[case] target: PType) {
        // Use values that fit in all target types (including i8: -128..127).
        let vec: PVector<u32> = buffer![0u32, 10, 100].into();
        let result = vec.cast(&target.into()).unwrap();
        assert!(result.as_primitive().validity().all_true());
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn cast_various_types_to_f64() {
        // Test casting from various primitive types to f64.
        let u8_vec: PVector<u8> = buffer![0u8, 1, 2, 3, 255].into();
        assert!(u8_vec.cast(&PType::F64.into()).is_ok());

        let u16_vec: PVector<u16> = buffer![0u16, 100, 1000].into();
        assert!(u16_vec.cast(&PType::F64.into()).is_ok());

        let u32_vec: PVector<u32> = buffer![0u32, 100, 1000, 1000000].into();
        assert!(u32_vec.cast(&PType::F64.into()).is_ok());

        let i8_vec: PVector<i8> = buffer![0i8, -1, 1, 127].into();
        assert!(i8_vec.cast(&PType::F64.into()).is_ok());

        let i32_vec: PVector<i32> = buffer![-1000000i32, -1, 0, 1, 1000000].into();
        assert!(i32_vec.cast(&PType::F64.into()).is_ok());

        let f32_vec: PVector<f32> = buffer![0.0f32, 1.5, -2.5, 100.0].into();
        assert!(f32_vec.cast(&PType::F64.into()).is_ok());
    }

    #[test]
    fn cast_u32_u8() {
        let vec: PVector<u32> = buffer![0u32, 10, 200].into();

        // Cast from u32 to u8.
        let result = vec.cast(&PType::U8.into()).unwrap();
        let p = result.into_primitive().into_u8();
        assert_eq!(p.as_ref(), &[0u8, 10, 200]);
        assert!(p.validity().all_true());
    }

    #[test]
    fn cast_u32_f32() {
        let vec: PVector<u32> = buffer![0u32, 10, 200].into();
        let result = vec.cast(&PType::F32.into()).unwrap();
        let p = result.into_primitive().into_f32();
        assert_eq!(p.as_ref(), &[0.0f32, 10., 200.]);
    }

    #[test]
    fn cast_i32_u32_overflow() {
        let vec: PVector<i32> = buffer![-1i32].into();
        let error = vec.cast(&PType::U32.into()).err().unwrap();
        let VortexError::ComputeError(s, _) = error else {
            unreachable!()
        };
        assert_eq!(s.to_string(), "Failed to cast -1 to U32");
    }

    #[test]
    fn cast_with_invalid_nulls() {
        // Create a vector with an invalid value at position 0 (which would overflow).
        let vec: PVector<i32> = PVector::new(
            buffer![-1i32, 0, 10],
            Mask::from(BitBuffer::from(vec![false, true, true])),
        );

        // Cast to nullable u32 should succeed because the invalid value is masked.
        let result = vec
            .cast(&DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap();
        let p = result.into_primitive().into_u32();
        assert_eq!(p.as_ref(), &[0u32, 0, 10]);
        assert_eq!(
            *p.validity(),
            Mask::from(BitBuffer::from(vec![false, true, true]))
        );
    }

    #[test]
    fn cast_all_null_vector() {
        let vec: PVector<i32> = PVector::new(buffer![-1i32, -2, -3], Mask::new_false(3));

        // Cast to nullable u32 should succeed because all values are masked.
        let result = vec
            .cast(&DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap();
        let p = result.into_primitive().into_u32();
        assert_eq!(p.as_ref(), &[0u32, 0, 0]);
        assert!(p.validity().all_false());
    }

    #[rstest]
    #[case(42i32, PType::U32)]
    #[case(0i32, PType::U8)]
    #[case(255i32, PType::U8)]
    #[case(100i32, PType::F64)]
    fn cast_scalar_valid(#[case] value: i32, #[case] target: PType) {
        let scalar: PScalar<i32> = PScalar::new(Some(value));
        let result = scalar.cast(&target.into()).unwrap();
        assert!(result.as_primitive().is_valid());
    }

    #[test]
    fn cast_scalar_i32_u32_overflow() {
        let scalar: PScalar<i32> = PScalar::new(Some(-1));
        let error = scalar.cast(&PType::U32.into()).err().unwrap();
        let VortexError::ComputeError(s, _) = error else {
            unreachable!()
        };
        assert_eq!(s.to_string(), "Failed to cast -1 to U32");
    }

    #[test]
    fn cast_scalar_null() {
        let scalar: PScalar<i32> = PScalar::null();
        let result = scalar
            .cast(&DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap();
        let p = result.into_primitive().into_u32();
        assert_eq!(p.value(), None);
    }

    #[test]
    fn cast_scalar_u32_f64() {
        let scalar: PScalar<u32> = PScalar::new(Some(12345));
        let result = scalar.cast(&PType::F64.into()).unwrap();
        let p = result.into_primitive().into_f64();
        assert_eq!(p.value(), Some(12345.0f64));
    }
}

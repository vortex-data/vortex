// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_dtype::{DType, NativePType, PType};
use vortex_error::{VortexResult, vortex_panic};

use crate::ToCanonical;
use crate::arrays::PrimitiveArray;
use crate::compute::{cast, min_max};
use crate::vtable::ValidityHelper;

impl PrimitiveArray {
    /// Return a slice of the array's buffer.
    ///
    /// NOTE: these values may be nonsense if the validity buffer indicates that the value is null.
    pub fn as_slice<T: NativePType>(&self) -> &[T] {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get slice of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }
        let raw_slice = self.byte_buffer().as_ptr();
        // SAFETY: alignment of Buffer is checked on construction
        unsafe {
            std::slice::from_raw_parts(raw_slice.cast(), self.byte_buffer().len() / size_of::<T>())
        }
    }

    pub fn reinterpret_cast(&self, ptype: PType) -> Self {
        if self.ptype() == ptype {
            return self.clone();
        }

        assert_eq!(
            self.ptype().byte_width(),
            ptype.byte_width(),
            "can't reinterpret cast between integers of two different widths"
        );

        PrimitiveArray::from_byte_buffer(self.byte_buffer().clone(), ptype, self.validity().clone())
    }

    /// Narrow the array to the smallest possible integer type that can represent all values,
    /// with a minimum byte size constraint.
    ///
    /// This method will narrow to the smallest type that can hold the values and has at least
    /// `min_size` bytes. For example, if values fit in U8 but `min_size` is 2, the result
    /// will be U16.
    ///
    /// # Arguments
    /// * `min_size` - Minimum byte width for the result type (0 means no constraint)
    pub fn narrow_min(&self, min_size: usize) -> VortexResult<PrimitiveArray> {
        if !self.ptype().is_int() {
            return Ok(self.clone());
        }

        let Some(min_max) = min_max(self.as_ref())? else {
            return Ok(PrimitiveArray::new(
                Buffer::<u8>::zeroed(self.len()),
                self.validity.clone(),
            ));
        };

        // If we can't cast to i64, then leave the array as its original type.
        // It's too big to downcast anyway.
        let Ok(min) = min_max.min.cast(&PType::I64.into()).and_then(i64::try_from) else {
            return Ok(self.clone());
        };
        let Ok(max) = min_max.max.cast(&PType::I64.into()).and_then(i64::try_from) else {
            return Ok(self.clone());
        };

        if min < 0 || max < 0 {
            // Signed
            if min >= i8::MIN as i64 && max <= i8::MAX as i64 && min_size <= 1 {
                return Ok(cast(
                    self.as_ref(),
                    &DType::Primitive(PType::I8, self.dtype().nullability()),
                )?
                .to_primitive());
            }

            if min >= i16::MIN as i64 && max <= i16::MAX as i64 && min_size <= 2 {
                return Ok(cast(
                    self.as_ref(),
                    &DType::Primitive(PType::I16, self.dtype().nullability()),
                )?
                .to_primitive());
            }

            if min >= i32::MIN as i64 && max <= i32::MAX as i64 && min_size <= 4 {
                return Ok(cast(
                    self.as_ref(),
                    &DType::Primitive(PType::I32, self.dtype().nullability()),
                )?
                .to_primitive());
            }
        } else {
            // Unsigned
            if max <= u8::MAX as i64 && min_size <= 1 {
                return Ok(cast(
                    self.as_ref(),
                    &DType::Primitive(PType::U8, self.dtype().nullability()),
                )?
                .to_primitive());
            }

            if max <= u16::MAX as i64 && min_size <= 2 {
                return Ok(cast(
                    self.as_ref(),
                    &DType::Primitive(PType::U16, self.dtype().nullability()),
                )?
                .to_primitive());
            }

            if max <= u32::MAX as i64 && min_size <= 4 {
                return Ok(cast(
                    self.as_ref(),
                    &DType::Primitive(PType::U32, self.dtype().nullability()),
                )?
                .to_primitive());
            }
        }

        Ok(self.clone())
    }

    /// Narrow the array to the smallest possible integer type that can represent all values.
    pub fn narrow(&self) -> VortexResult<PrimitiveArray> {
        self.narrow_min(0)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::{Buffer, buffer};
    use vortex_dtype::{DType, Nullability, PType};

    use crate::arrays::PrimitiveArray;
    use crate::validity::Validity;

    #[test]
    fn test_downcast_all_invalid() {
        let array = PrimitiveArray::new(
            buffer![0_u32, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            Validity::AllInvalid,
        );

        let result = array.narrow().unwrap();
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::U8, Nullability::Nullable)
        );
        assert_eq!(result.validity, Validity::AllInvalid);
    }

    #[rstest]
    #[case(vec![0_i64, 127], PType::U8)]
    #[case(vec![-128_i64, 127], PType::I8)]
    #[case(vec![-129_i64, 127], PType::I16)]
    #[case(vec![-128_i64, 128], PType::I16)]
    #[case(vec![-32768_i64, 32767], PType::I16)]
    #[case(vec![-32769_i64, 32767], PType::I32)]
    #[case(vec![-32768_i64, 32768], PType::I32)]
    #[case(vec![i32::MIN as i64, i32::MAX as i64], PType::I32)]
    fn test_downcast_signed(#[case] values: Vec<i64>, #[case] expected_ptype: PType) {
        let array = PrimitiveArray::from_iter(values);
        let result = array.narrow().unwrap();
        assert_eq!(result.ptype(), expected_ptype);
    }

    #[rstest]
    #[case(vec![0_u64, 255], PType::U8)]
    #[case(vec![0_u64, 256], PType::U16)]
    #[case(vec![0_u64, 65535], PType::U16)]
    #[case(vec![0_u64, 65536], PType::U32)]
    #[case(vec![0_u64, u32::MAX as u64], PType::U32)]
    fn test_downcast_unsigned(#[case] values: Vec<u64>, #[case] expected_ptype: PType) {
        let array = PrimitiveArray::from_iter(values);
        let result = array.narrow().unwrap();
        assert_eq!(result.ptype(), expected_ptype);
    }

    #[test]
    fn test_downcast_keeps_original_if_too_large() {
        let array = PrimitiveArray::from_iter(vec![0_u64, u64::MAX]);
        let result = array.narrow().unwrap();
        assert_eq!(result.ptype(), PType::U64);
    }

    #[test]
    fn test_downcast_preserves_nullability() {
        let array = PrimitiveArray::from_option_iter([Some(0_i32), None, Some(127)]);
        let result = array.narrow().unwrap();
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::U8, Nullability::Nullable)
        );
        // Check that validity is preserved (the array should still have nullable values)
        assert!(matches!(&result.validity, Validity::Array(_)));
    }

    #[test]
    fn test_downcast_preserves_values() {
        let values = vec![-100_i16, 0, 100];
        let array = PrimitiveArray::from_iter(values);
        let result = array.narrow().unwrap();

        assert_eq!(result.ptype(), PType::I8);
        // Check that the values were properly downscaled
        let downscaled_values: Vec<i8> = result.as_slice::<i8>().to_vec();
        assert_eq!(downscaled_values, vec![-100_i8, 0, 100]);
    }

    #[test]
    fn test_downcast_with_mixed_signs_chooses_signed() {
        let array = PrimitiveArray::from_iter(vec![-1_i32, 200]);
        let result = array.narrow().unwrap();
        assert_eq!(result.ptype(), PType::I16);
    }

    #[test]
    fn test_downcast_floats() {
        let array = PrimitiveArray::from_iter(vec![1.0_f32, 2.0, 3.0]);
        let result = array.narrow().unwrap();
        // Floats should remain unchanged since they can't be downscaled to integers
        assert_eq!(result.ptype(), PType::F32);
    }

    #[test]
    fn test_downcast_empty_array() {
        let array = PrimitiveArray::new(Buffer::<i32>::empty(), Validity::AllInvalid);
        let result = array.narrow().unwrap();
        let array2 = PrimitiveArray::new(Buffer::<i64>::empty(), Validity::NonNullable);
        let result2 = array2.narrow().unwrap();
        // Empty arrays should not have their validity changed
        assert_eq!(result.validity, Validity::AllInvalid);
        assert_eq!(result2.validity, Validity::NonNullable);
    }

    #[rstest]
    #[case(vec![0_i64, 255], 0, PType::U8)] // No constraint, fits in U8
    #[case(vec![0_i64, 255], 1, PType::U8)] // Min 1 byte, fits in U8
    #[case(vec![0_i64, 255], 2, PType::U16)] // Min 2 bytes, needs U16
    #[case(vec![0_i64, 255], 4, PType::U32)] // Min 4 bytes, needs U32
    #[case(vec![0_i64, 100], 2, PType::U16)] // Min 2 bytes, even though values fit in U8
    #[case(vec![0_i64, 127], 0, PType::U8)] // No constraint, unsigned U8
    #[case(vec![-1_i64, 127], 0, PType::I8)] // No constraint, signed I8
    #[case(vec![-1_i64, 127], 2, PType::I16)] // Min 2 bytes, needs I16
    #[case(vec![0_i64, 256], 1, PType::U16)] // Naturally needs U16, min 1 byte
    #[case(vec![0_i64, 256], 2, PType::U16)] // Naturally needs U16, min 2 bytes
    fn test_narrow_min(
        #[case] values: Vec<i64>,
        #[case] min_size: usize,
        #[case] expected_ptype: PType,
    ) {
        let array = PrimitiveArray::from_iter(values);
        let result = array.narrow_min(min_size).unwrap();
        assert_eq!(
            result.ptype(),
            expected_ptype,
            "narrow_min({}) failed",
            min_size
        );
    }

    #[test]
    fn test_narrow_min_preserves_values() {
        // Values fit in U8 but we request minimum 2 bytes
        let values = vec![0_u32, 100, 200];
        let array = PrimitiveArray::from_iter(values);
        let result = array.narrow_min(2).unwrap();

        // Should be U16
        assert_eq!(result.ptype(), PType::U16);
        // Values should be preserved
        let result_values: Vec<u16> = result.as_slice::<u16>().to_vec();
        assert_eq!(result_values, vec![0_u16, 100, 200]);
    }

    #[test]
    fn test_narrow_min_signed() {
        // Values fit in I8 but we request minimum 4 bytes
        let values = vec![-50_i64, 0, 50];
        let array = PrimitiveArray::from_iter(values);
        let result = array.narrow_min(4).unwrap();

        // Should be I32
        assert_eq!(result.ptype(), PType::I32);
        let result_values: Vec<i32> = result.as_slice::<i32>().to_vec();
        assert_eq!(result_values, vec![-50_i32, 0, 50]);
    }

    #[test]
    fn test_narrow_min_respects_natural_size() {
        // Values naturally need U32, requesting min 2 bytes shouldn't downgrade
        let values = vec![0_u64, u32::MAX as u64];
        let array = PrimitiveArray::from_iter(values);
        let result = array.narrow_min(2).unwrap();

        // Should stay U32 (can't fit in U16)
        assert_eq!(result.ptype(), PType::U32);
    }
}

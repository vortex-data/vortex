// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_dtype::{DType, PType};
use vortex_error::VortexResult;

use crate::ToCanonical;
use crate::arrays::PrimitiveArray;
use crate::compute::{cast, min_max};
use crate::validity::Validity;

impl PrimitiveArray {
    pub fn downcast(&self) -> VortexResult<PrimitiveArray> {
        if !self.ptype().is_int() {
            return Ok(self.clone());
        }

        let Some(min_max) = min_max(self.as_ref())? else {
            return Ok(PrimitiveArray::new(
                Buffer::<u8>::zeroed(self.len()),
                Validity::AllInvalid,
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
            if min >= i8::MIN as i64 && max <= i8::MAX as i64 {
                return cast(
                    self.as_ref(),
                    &DType::Primitive(PType::I8, self.dtype().nullability()),
                )?
                .to_primitive();
            }

            if min >= i16::MIN as i64 && max <= i16::MAX as i64 {
                return cast(
                    self.as_ref(),
                    &DType::Primitive(PType::I16, self.dtype().nullability()),
                )?
                .to_primitive();
            }

            if min >= i32::MIN as i64 && max <= i32::MAX as i64 {
                return cast(
                    self.as_ref(),
                    &DType::Primitive(PType::I32, self.dtype().nullability()),
                )?
                .to_primitive();
            }
        } else {
            // Unsigned
            if max <= u8::MAX as i64 {
                return cast(
                    self.as_ref(),
                    &DType::Primitive(PType::U8, self.dtype().nullability()),
                )?
                .to_primitive();
            }

            if max <= u16::MAX as i64 {
                return cast(
                    self.as_ref(),
                    &DType::Primitive(PType::U16, self.dtype().nullability()),
                )?
                .to_primitive();
            }

            if max <= u32::MAX as i64 {
                return cast(
                    self.as_ref(),
                    &DType::Primitive(PType::U32, self.dtype().nullability()),
                )?
                .to_primitive();
            }
        }

        Ok(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::arrays::PrimitiveArray;
    use crate::validity::Validity;

    #[test]
    fn test_downcast_all_invalid() {
        let array = PrimitiveArray::new(
            buffer![0_u32, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            Validity::AllInvalid,
        );

        let result = array.downcast().unwrap();
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
        let result = array.downcast().unwrap();
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
        let result = array.downcast().unwrap();
        assert_eq!(result.ptype(), expected_ptype);
    }

    #[test]
    fn test_downcast_keeps_original_if_too_large() {
        let array = PrimitiveArray::from_iter(vec![0_u64, u64::MAX]);
        let result = array.downcast().unwrap();
        assert_eq!(result.ptype(), PType::U64);
    }

    #[test]
    fn test_downcast_preserves_nullability() {
        let array = PrimitiveArray::from_option_iter([Some(0_i32), None, Some(127)]);
        let result = array.downcast().unwrap();
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
        let result = array.downcast().unwrap();

        assert_eq!(result.ptype(), PType::I8);
        // Check that the values were properly downscaled
        let downscaled_values: Vec<i8> = result.as_slice::<i8>().to_vec();
        assert_eq!(downscaled_values, vec![-100_i8, 0, 100]);
    }

    #[test]
    fn test_downcast_with_mixed_signs_chooses_signed() {
        let array = PrimitiveArray::from_iter(vec![-1_i32, 200]);
        let result = array.downcast().unwrap();
        assert_eq!(result.ptype(), PType::I16);
    }

    #[test]
    fn test_downcast_floats() {
        let array = PrimitiveArray::from_iter(vec![1.0_f32, 2.0, 3.0]);
        let result = array.downcast().unwrap();
        // Floats should remain unchanged since they can't be downscaled to integers
        assert_eq!(result.ptype(), PType::F32);
    }

    #[test]
    fn test_downcast_empty_array() {
        let array = PrimitiveArray::from_iter(Vec::<i32>::new());
        let result = array.downcast().unwrap();
        // Empty arrays should get all invalid buffer
        assert_eq!(result.validity, Validity::AllInvalid);
    }
}

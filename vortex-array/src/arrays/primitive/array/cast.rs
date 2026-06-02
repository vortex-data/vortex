// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use super::PrimitiveData;
use crate::dtype::NativePType;

impl PrimitiveData {
    /// Return a slice of the array's buffer.
    ///
    /// NOTE: these values may be nonsense if the validity buffer indicates that the value is null.
    ///
    /// # Panic
    ///
    /// This operation will panic if the array is not backed by host memory.
    pub fn as_slice<T: NativePType>(&self) -> &[T] {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get slice of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }

        let byte_buffer = self
            .buffer
            .as_host_opt()
            .vortex_expect("as_slice must be called on host buffer");
        let raw_slice = byte_buffer.as_ptr();

        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts(raw_slice.cast(), byte_buffer.len() / size_of::<T>()) }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::primitive::PrimitiveArrayExt;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::session::ArraySession;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_downcast_all_invalid() {
        let array = PrimitiveArray::new(
            buffer![0_u32, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            Validity::AllInvalid,
        );

        let result = array.narrow(&mut SESSION.create_execution_ctx()).unwrap();
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::U8, Nullability::Nullable)
        );
        assert!(matches!(result.validity(), Ok(Validity::AllInvalid)));
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
        let result = array.narrow(&mut SESSION.create_execution_ctx()).unwrap();
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
        let result = array.narrow(&mut SESSION.create_execution_ctx()).unwrap();
        assert_eq!(result.ptype(), expected_ptype);
    }

    #[test]
    fn test_downcast_keeps_original_if_too_large() {
        let array = PrimitiveArray::from_iter(vec![0_u64, u64::MAX]);
        let result = array.narrow(&mut SESSION.create_execution_ctx()).unwrap();
        assert_eq!(result.ptype(), PType::U64);
    }

    #[test]
    fn test_downcast_preserves_nullability() {
        let array = PrimitiveArray::from_option_iter([Some(0_i32), None, Some(127)]);
        let result = array.narrow(&mut SESSION.create_execution_ctx()).unwrap();
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::U8, Nullability::Nullable)
        );
        // Check that validity is preserved (the array should still have nullable values)
        assert!(matches!(result.validity(), Ok(Validity::Array(_))));
    }

    #[test]
    fn test_downcast_preserves_values() {
        let values = vec![-100_i16, 0, 100];
        let array = PrimitiveArray::from_iter(values);
        let result = array.narrow(&mut SESSION.create_execution_ctx()).unwrap();

        assert_eq!(result.ptype(), PType::I8);
        // Check that the values were properly downscaled
        let downscaled_values: Vec<i8> = result.as_slice::<i8>().to_vec();
        assert_eq!(downscaled_values, vec![-100_i8, 0, 100]);
    }

    #[test]
    fn test_downcast_with_mixed_signs_chooses_signed() {
        let array = PrimitiveArray::from_iter(vec![-1_i32, 200]);
        let result = array.narrow(&mut SESSION.create_execution_ctx()).unwrap();
        assert_eq!(result.ptype(), PType::I16);
    }

    #[test]
    fn test_downcast_floats() {
        let array = PrimitiveArray::from_iter(vec![1.0_f32, 2.0, 3.0]);
        let result = array.narrow(&mut SESSION.create_execution_ctx()).unwrap();
        // Floats should remain unchanged since they can't be downscaled to integers
        assert_eq!(result.ptype(), PType::F32);
    }

    #[test]
    fn test_downcast_empty_array() {
        let array = PrimitiveArray::new(Buffer::<i32>::empty(), Validity::AllInvalid);
        let result = array.narrow(&mut SESSION.create_execution_ctx()).unwrap();
        let array2 = PrimitiveArray::new(Buffer::<i64>::empty(), Validity::NonNullable);
        let result2 = array2.narrow(&mut SESSION.create_execution_ctx()).unwrap();
        // Empty arrays should not have their validity changed
        assert!(matches!(result.validity(), Ok(Validity::AllInvalid)));
        assert!(matches!(result2.validity(), Ok(Validity::NonNullable)));
    }
}

mod array;
pub mod compute;
mod ops;
mod patch;
mod serde;

pub use array::*;
// Re-export the BooleanBuffer type on our API surface.
pub use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};

#[cfg(test)]
mod tests {
    use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
    use vortex_buffer::buffer;

    use crate::ToCanonical;
    use crate::array::Array;
    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::compute::conformance::mask::test_mask;
    use crate::patches::Patches;
    use crate::validity::Validity;

    #[test]
    fn bool_array() {
        let arr = BoolArray::from_iter([true, false, true]);
        let scalar = bool::try_from(&arr.scalar_at(0).unwrap()).unwrap();
        assert!(scalar);
    }

    #[test]
    fn test_all_some_iter() {
        let arr = BoolArray::from_iter([Some(true), Some(false)]);

        assert!(matches!(arr.validity(), Validity::AllValid));

        let scalar = bool::try_from(&arr.scalar_at(0).unwrap()).unwrap();
        assert!(scalar);
        let scalar = bool::try_from(&arr.scalar_at(1).unwrap()).unwrap();
        assert!(!scalar);
    }

    #[test]
    fn test_bool_from_iter() {
        let arr = BoolArray::from_iter([Some(true), Some(true), None, Some(false), None]);

        let scalar = bool::try_from(&arr.scalar_at(0).unwrap()).unwrap();
        assert!(scalar);

        let scalar = bool::try_from(&arr.scalar_at(1).unwrap()).unwrap();
        assert!(scalar);

        let scalar = arr.scalar_at(2).unwrap();
        assert!(scalar.is_null());

        let scalar = bool::try_from(&arr.scalar_at(3).unwrap()).unwrap();
        assert!(!scalar);

        let scalar = arr.scalar_at(4).unwrap();
        assert!(scalar.is_null());
    }

    #[test]
    fn patch_sliced_bools() {
        let arr = {
            let mut builder = BooleanBufferBuilder::new(12);
            builder.append(false);
            builder.append_n(11, true);
            BoolArray::from(builder.finish())
        };
        let sliced = arr.slice(4, 12).unwrap();
        let sliced_len = sliced.len();
        let (values, offset) = sliced.to_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.as_slice(), &[254, 15]);

        // patch the underlying array
        let patches = Patches::new(
            arr.len(),
            0,
            PrimitiveArray::new(buffer![4u32], Validity::AllValid).into_array(),
            BoolArray::from(BooleanBuffer::new_unset(1)).into_array(),
        );
        let arr = arr.patch(&patches).unwrap();
        let arr_len = arr.len();
        let (values, offset) = arr.to_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 0);
        assert_eq!(values.len(), arr_len + offset);
        assert_eq!(values.as_slice(), &[238, 15]);

        // the slice should be unchanged
        let (values, offset) = sliced.to_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.len(), sliced_len + offset);
        assert_eq!(values.as_slice(), &[254, 15]); // unchanged
    }

    #[test]
    fn slice_array_in_middle() {
        let arr = BoolArray::from(BooleanBuffer::new_set(16));
        let sliced = arr.slice(4, 12).unwrap();
        let sliced_len = sliced.len();
        let (values, offset) = sliced.to_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.len(), sliced_len + offset);
        assert_eq!(values.as_slice(), &[255, 15]);
    }

    #[test]
    #[should_panic]
    fn patch_bools_owned() {
        let buffer = buffer![255u8; 2];
        let buf = BooleanBuffer::new(buffer.into_arrow_buffer(), 0, 15);
        let arr = BoolArray::new(buf, Validity::NonNullable);
        let buf_ptr = arr.boolean_buffer().sliced().as_ptr();

        let patches = Patches::new(
            arr.len(),
            0,
            PrimitiveArray::new(buffer![0u32], Validity::AllValid).into_array(),
            BoolArray::from(BooleanBuffer::new_unset(1)).into_array(),
        );
        let arr = arr.patch(&patches).unwrap();
        assert_eq!(arr.boolean_buffer().sliced().as_ptr(), buf_ptr);

        let (values, _byte_bit_offset) = arr.to_bool().unwrap().into_boolean_builder();
        assert_eq!(values.as_slice(), &[254, 127]);
    }

    #[test]
    fn test_mask_primitive_array() {
        test_mask(&BoolArray::from_iter([true, false, true, true, false]));
    }
}

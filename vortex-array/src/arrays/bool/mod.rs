use arrow_array::BooleanArray;
use arrow_buffer::MutableBuffer;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::VortexExpect;

use crate::validity::Validity;
use crate::{Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayVisitorImpl, RkyvMetadata};

mod array;
pub mod compute;
mod patch;

pub use array::*;
// Re-export the BooleanBuffer type on our API surface.
pub use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};

impl BoolArray {
    /// Create a new BoolArray from a set of indices and a length.
    /// All indices must be less than the length.
    pub fn from_indices<I: IntoIterator<Item = usize>>(length: usize, indices: I) -> Self {
        let mut buffer = MutableBuffer::new_null(length);
        indices
            .into_iter()
            .for_each(|idx| arrow_buffer::bit_util::set_bit(&mut buffer, idx));
        Self::new(
            BooleanBufferBuilder::new_from_buffer(buffer, length).finish(),
            Validity::NonNullable,
        )
    }
}

impl From<BooleanBuffer> for BoolArray {
    fn from(value: BooleanBuffer) -> Self {
        Self::new(value, Validity::NonNullable)
    }
}

impl FromIterator<bool> for BoolArray {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        Self::new(BooleanBuffer::from_iter(iter), Validity::NonNullable)
    }
}

impl FromIterator<Option<bool>> for BoolArray {
    fn from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        let (buffer, nulls) = BooleanArray::from_iter(iter).into_parts();

        Self::new(
            buffer,
            nulls.map(Validity::from).unwrap_or(Validity::AllValid),
        )
    }
}

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BoolMetadata {
    // We know the offset in bits must be <8
    offset: u8,
}

impl ArrayVisitorImpl<RkyvMetadata<BoolMetadata>> for BoolArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&ByteBuffer::from_arrow_buffer(
            self.boolean_buffer().clone().into_inner(),
            Alignment::none(),
        ))
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&self.validity, self.len());
    }

    fn _metadata(&self) -> RkyvMetadata<BoolMetadata> {
        let bit_offset = self.boolean_buffer().offset();
        assert!(bit_offset < 8, "Offset must be <8, got {}", bit_offset);
        RkyvMetadata(BoolMetadata {
            offset: u8::try_from(bit_offset).vortex_expect("checked"),
        })
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
    use vortex_buffer::buffer;

    use crate::ToCanonical;
    use crate::array::Array;
    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::compute::test_harness::test_mask;
    use crate::compute::{scalar_at, slice};
    use crate::patches::Patches;
    use crate::validity::Validity;

    #[test]
    fn bool_array() {
        let arr = BoolArray::from_iter([true, false, true]);
        let scalar = bool::try_from(&scalar_at(&arr, 0).unwrap()).unwrap();
        assert!(scalar);
    }

    #[test]
    fn test_all_some_iter() {
        let arr = BoolArray::from_iter([Some(true), Some(false)]);

        assert!(matches!(arr.validity(), Validity::AllValid));

        let scalar = bool::try_from(&scalar_at(&arr, 0).unwrap()).unwrap();
        assert!(scalar);
        let scalar = bool::try_from(&scalar_at(&arr, 1).unwrap()).unwrap();
        assert!(!scalar);
    }

    #[test]
    fn test_bool_from_iter() {
        let arr = BoolArray::from_iter([Some(true), Some(true), None, Some(false), None]);

        let scalar = bool::try_from(&scalar_at(&arr, 0).unwrap()).unwrap();
        assert!(scalar);

        let scalar = bool::try_from(&scalar_at(&arr, 1).unwrap()).unwrap();
        assert!(scalar);

        let scalar = scalar_at(&arr, 2).unwrap();
        assert!(scalar.is_null());

        let scalar = bool::try_from(&scalar_at(&arr, 3).unwrap()).unwrap();
        assert!(!scalar);

        let scalar = scalar_at(&arr, 4).unwrap();
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
        let sliced = slice(&arr, 4, 12).unwrap();
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
        let sliced = slice(&arr, 4, 12).unwrap();
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

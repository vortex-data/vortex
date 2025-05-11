use arrow_array::BooleanArray;
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, MutableBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_panic};

use crate::Canonical;
use crate::arrays::{BoolVTable, bool};
use crate::builders::ArrayBuilder;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::validity::Validity;
use crate::vtable::{ArrayVTable, CanonicalVTable, ValidityHelper};

#[derive(Clone, Debug)]
pub struct BoolArray {
    dtype: DType,
    buffer: BooleanBuffer,
    pub(crate) validity: Validity,
    pub(crate) stats_set: ArrayStats,
}

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

    /// Creates a new [`BoolArray`] from a [`BooleanBuffer`] and [`Validity`], without checking
    /// any invariants.
    pub fn new(buffer: BooleanBuffer, validity: Validity) -> Self {
        if let Some(len) = validity.maybe_len() {
            if buffer.len() != len {
                vortex_panic!(
                    "Buffer and validity length mismatch: buffer={}, validity={}",
                    buffer.len(),
                    len
                );
            }
        }

        // Shrink the buffer to remove any whole bytes.
        let buffer = buffer.shrink_offset();
        Self {
            dtype: DType::Bool(validity.nullability()),
            buffer,
            validity,
            stats_set: ArrayStats::default(),
        }
    }

    /// Returns the underlying [`BooleanBuffer`] of the array.
    pub fn boolean_buffer(&self) -> &BooleanBuffer {
        assert!(
            self.buffer.offset() < 8,
            "Offset must be <8, did we forget to call shrink_offset? Found {}",
            self.buffer.offset()
        );
        &self.buffer
    }

    /// Get a mutable version of this array.
    ///
    /// If the caller holds the only reference to the underlying buffer the underlying buffer is returned
    /// otherwise a copy is created.
    ///
    /// The second value of the tuple is a bit_offset of first value in first byte of the returned builder
    pub fn into_boolean_builder(self) -> (BooleanBufferBuilder, usize) {
        let offset = self.buffer.offset();
        let len = self.buffer.len();
        let arrow_buffer = self.buffer.into_inner();
        let mutable_buf = if arrow_buffer.ptr_offset() == 0 {
            arrow_buffer.into_mutable().unwrap_or_else(|b| {
                let mut buf = MutableBuffer::with_capacity(b.len());
                buf.extend_from_slice(b.as_slice());
                buf
            })
        } else {
            let mut buf = MutableBuffer::with_capacity(arrow_buffer.len());
            buf.extend_from_slice(arrow_buffer.as_slice());
            buf
        };

        (
            BooleanBufferBuilder::new_from_buffer(mutable_buf, offset + len),
            offset,
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

impl ValidityHelper for BoolArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}

impl ArrayVTable<BoolVTable> for BoolVTable {
    fn len(array: &BoolArray) -> usize {
        array.buffer.len()
    }

    fn dtype(array: &BoolArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &BoolArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<BoolVTable> for BoolVTable {
    fn canonicalize(array: &BoolArray) -> VortexResult<Canonical> {
        Ok(Canonical::Bool(array.clone()))
    }

    fn append_to_builder(array: &BoolArray, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        builder.extend_from_array(array.as_ref())
    }
}

pub trait BooleanBufferExt {
    /// Slice any full bytes from the buffer, leaving the offset < 8.
    fn shrink_offset(self) -> Self;
}

impl BooleanBufferExt for BooleanBuffer {
    fn shrink_offset(self) -> Self {
        let byte_offset = self.offset() / 8;
        let bit_offset = self.offset() % 8;
        let len = self.len();
        let buffer = self
            .into_inner()
            .slice_with_length(byte_offset, (len + bit_offset).div_ceil(8));
        BooleanBuffer::new(buffer, bit_offset, len)
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
    use vortex_buffer::buffer;

    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::compute::conformance::mask::test_mask;
    use crate::patches::Patches;
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;
    use crate::{Array, IntoArray, ToCanonical};

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
        test_mask(BoolArray::from_iter([true, false, true, true, false]).as_ref());
    }
}

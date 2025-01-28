use std::fmt::{Debug, Display};

use arrow_array::BooleanArray;
use arrow_buffer::MutableBuffer;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};

use crate::encoding::ids;
use crate::stats::StatsSet;
use crate::validate::ValidateVTable;
use crate::validity::{LogicalValidity, Validity, ValidityMetadata, ValidityVTable};
use crate::variants::{BoolArrayTrait, VariantsVTable};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{
    impl_encoding, ArrayLen, Canonical, DeserializeMetadata, IntoArrayData, IntoCanonical,
    RkyvMetadata,
};

pub mod compute;
mod patch;
mod stats;

// Re-export the BooleanBuffer type on our API surface.
pub use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};

impl_encoding!("vortex.bool", ids::BOOL, Bool, RkyvMetadata<BoolMetadata>);

#[derive(
    Clone,
    Debug,
    rkyv::Archive,
    rkyv::Portable,
    rkyv::Serialize,
    rkyv::Deserialize,
    rkyv::bytecheck::CheckBytes,
)]
#[bytecheck(crate = rkyv::bytecheck)]
#[repr(C)]
pub struct BoolMetadata {
    pub(crate) validity: ValidityMetadata,
    pub(crate) first_byte_bit_offset: u8,
}

impl Display for BoolMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl BoolArray {
    /// Access internal array buffer
    pub fn buffer(&self) -> &ByteBuffer {
        self.as_ref()
            .byte_buffer(0)
            .vortex_expect("Missing buffer in BoolArray")
    }

    /// Convert array into its internal buffer
    pub fn into_buffer(self) -> ByteBuffer {
        self.into_array()
            .into_byte_buffer(0)
            .vortex_expect("BoolArray must have a buffer")
    }

    /// Get array values as an arrow [BooleanBuffer]
    pub fn boolean_buffer(&self) -> BooleanBuffer {
        BooleanBuffer::new(
            self.buffer().clone().into_arrow_buffer(),
            self.metadata().first_byte_bit_offset as usize,
            self.len(),
        )
    }

    /// Get a mutable version of this array.
    ///
    /// If the caller holds the only reference to the underlying buffer the underlying buffer is returned
    /// otherwise a copy is created.
    ///
    /// The second value of the tuple is a bit_offset of first value in first byte of the returned builder
    pub fn into_boolean_builder(self) -> (BooleanBufferBuilder, usize) {
        let first_byte_bit_offset = self.metadata().first_byte_bit_offset as usize;
        let len = self.len();
        let arrow_buffer = self.into_buffer().into_arrow_buffer();
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
            BooleanBufferBuilder::new_from_buffer(mutable_buf, len + first_byte_bit_offset),
            first_byte_bit_offset,
        )
    }

    pub fn validity(&self) -> Validity {
        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(0, &Validity::DTYPE, self.len())
                .vortex_expect("BoolArray: validity child")
        })
    }

    /// Create a new BoolArray from a buffer and nullability.
    pub fn new(buffer: BooleanBuffer, nullability: Nullability) -> Self {
        let validity = match nullability {
            Nullability::Nullable => Validity::AllValid,
            Nullability::NonNullable => Validity::NonNullable,
        };
        Self::try_new(buffer, validity).vortex_expect("Validity length cannot be mismatched")
    }

    /// Create a new BoolArray from a buffer and validity metadata.
    /// Returns an error if the validity length does not match the buffer length.
    #[allow(clippy::cast_possible_truncation)]
    pub fn try_new(buffer: BooleanBuffer, validity: Validity) -> VortexResult<Self> {
        let buffer_len = buffer.len();
        let buffer_offset = buffer.offset();
        let first_byte_bit_offset = (buffer_offset % 8) as u8;
        let buffer_byte_offset = buffer_offset - (first_byte_bit_offset as usize);

        let inner = buffer
            .into_inner()
            .bit_slice(buffer_byte_offset, buffer_len);

        Self::try_from_parts(
            DType::Bool(validity.nullability()),
            buffer_len,
            RkyvMetadata(BoolMetadata {
                validity: validity.to_metadata(buffer_len)?,
                first_byte_bit_offset,
            }),
            Some(vec![ByteBuffer::from_arrow_buffer(inner, Alignment::of::<u8>())].into()),
            validity.into_array().map(|v| [v].into()),
            StatsSet::default(),
        )
    }

    /// Create a new BoolArray from a set of indices and a length.
    /// All indices must be less than the length.
    pub fn from_indices<I: IntoIterator<Item = usize>>(length: usize, indices: I) -> Self {
        let mut buffer = MutableBuffer::new_null(length);
        indices
            .into_iter()
            .for_each(|idx| arrow_buffer::bit_util::set_bit(&mut buffer, idx));
        Self::new(
            BooleanBufferBuilder::new_from_buffer(buffer, length).finish(),
            Nullability::NonNullable,
        )
    }
}

impl ValidateVTable<BoolArray> for BoolEncoding {
    fn validate(&self, array: &BoolArray) -> VortexResult<()> {
        if array.as_ref().nbuffers() != 1 {
            vortex_bail!(
                "BoolArray: expected 1 buffer, found {}",
                array.as_ref().nbuffers()
            );
        }

        Ok(())
    }
}

impl VariantsVTable<BoolArray> for BoolEncoding {
    fn as_bool_array<'a>(&self, array: &'a BoolArray) -> Option<&'a dyn BoolArrayTrait> {
        Some(array)
    }
}

impl BoolArrayTrait for BoolArray {}

impl From<BooleanBuffer> for BoolArray {
    fn from(value: BooleanBuffer) -> Self {
        Self::new(value, Nullability::NonNullable)
    }
}

impl FromIterator<bool> for BoolArray {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        Self::new(BooleanBuffer::from_iter(iter), Nullability::NonNullable)
    }
}

impl FromIterator<Option<bool>> for BoolArray {
    fn from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        let (buffer, nulls) = BooleanArray::from_iter(iter).into_parts();
        Self::try_new(
            buffer,
            nulls.map(Validity::from).unwrap_or(Validity::AllValid),
        )
        .vortex_expect("Validity length cannot be mismatched")
    }
}

impl IntoCanonical for BoolArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        Ok(Canonical::Bool(self))
    }
}

impl ValidityVTable<BoolArray> for BoolEncoding {
    fn is_valid(&self, array: &BoolArray, index: usize) -> VortexResult<bool> {
        array.validity().is_valid(index)
    }

    fn logical_validity(&self, array: &BoolArray) -> VortexResult<LogicalValidity> {
        array.validity().to_logical(array.len())
    }
}

impl VisitorVTable<BoolArray> for BoolEncoding {
    fn accept(&self, array: &BoolArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_buffer(array.buffer())?;
        visitor.visit_validity(&array.validity())
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;

    use crate::array::{BoolArray, PrimitiveArray};
    use crate::compute::{scalar_at, slice};
    use crate::patches::Patches;
    use crate::validity::Validity;
    use crate::{ArrayLen, IntoArrayData, IntoArrayVariant};

    #[test]
    fn bool_array() {
        let arr = BoolArray::from_iter([true, false, true]).into_array();
        let scalar = bool::try_from(&scalar_at(&arr, 0).unwrap()).unwrap();
        assert!(scalar);
    }

    #[test]
    fn test_all_some_iter() {
        let arr = BoolArray::from_iter([Some(true), Some(false)]);

        assert!(matches!(arr.validity(), Validity::AllValid));

        let arr = arr.into_array();

        let scalar = bool::try_from(&scalar_at(&arr, 0).unwrap()).unwrap();
        assert!(scalar);
        let scalar = bool::try_from(&scalar_at(&arr, 1).unwrap()).unwrap();
        assert!(!scalar);
    }

    #[test]
    fn test_bool_from_iter() {
        let arr =
            BoolArray::from_iter([Some(true), Some(true), None, Some(false), None]).into_array();

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
        let sliced = slice(arr.clone(), 4, 12).unwrap();
        let (values, offset) = sliced.clone().into_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.as_slice(), &[254, 15]);

        // patch the underlying array
        let patches = Patches::new(
            arr.len(),
            PrimitiveArray::new(buffer![4u32], Validity::AllValid).into_array(),
            BoolArray::from(BooleanBuffer::new_unset(1)).into_array(),
        );
        let arr = arr.patch(patches).unwrap();
        let (values, offset) = arr.into_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 0);
        assert_eq!(values.as_slice(), &[238, 15]);

        // the slice should be unchanged
        let (values, offset) = sliced.into_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.as_slice(), &[254, 15]); // unchanged
    }

    #[test]
    #[should_panic]
    fn patch_bools_owned() {
        let buffer = buffer![255u8; 2];
        let buf = BooleanBuffer::new(buffer.into_arrow_buffer(), 0, 15);
        let arr = BoolArray::new(buf, Nullability::NonNullable);
        let buf_ptr = arr.boolean_buffer().sliced().as_ptr();

        let patches = Patches::new(
            arr.len(),
            PrimitiveArray::new(buffer![0u32], Validity::AllValid).into_array(),
            BoolArray::from(BooleanBuffer::new_unset(1)).into_array(),
        );
        let arr = arr.patch(patches).unwrap();
        assert_eq!(arr.boolean_buffer().sliced().as_ptr(), buf_ptr);

        let (values, offset) = arr.into_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 0);
        assert_eq!(values.as_slice(), &[254, 127]);
    }
}

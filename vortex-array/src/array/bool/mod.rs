use std::fmt::{Debug, Display};
use std::sync::Arc;

use arrow_array::BooleanArray;
use arrow_buffer::{BooleanBufferBuilder, MutableBuffer};
use serde::{Deserialize, Serialize};
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect as _, VortexResult};

use crate::encoding::ids;
use crate::stats::StatsSet;
use crate::validity::{LogicalValidity, Validity, ValidityMetadata, ValidityVTable};
use crate::variants::{BoolArrayTrait, VariantsVTable};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{
    impl_encoding, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoArrayData, IntoCanonical,
};

pub mod compute;
mod patch;
mod stats;

// Re-export the BooleanBuffer type on our API surface.
pub use arrow_buffer::BooleanBuffer;

impl_encoding!("vortex.bool", ids::BOOL, Bool);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoolMetadata {
    validity: ValidityMetadata,
    first_byte_bit_offset: u8,
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
            .byte_buffer()
            .vortex_expect("Missing buffer in BoolArray")
    }

    /// Convert array into its internal buffer
    pub fn into_buffer(self) -> ByteBuffer {
        self.into_array()
            .into_byte_buffer()
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

        ArrayData::try_new_owned(
            &BoolEncoding,
            DType::Bool(validity.nullability()),
            buffer_len,
            Arc::new(BoolMetadata {
                validity: validity.to_metadata(buffer_len)?,
                first_byte_bit_offset,
            }),
            Some(ByteBuffer::from_arrow_buffer(inner, Alignment::of::<u8>())),
            validity.into_array().into_iter().collect(),
            StatsSet::default(),
        )?
        .try_into()
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

impl ArrayTrait for BoolArray {}

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
    fn is_valid(&self, array: &BoolArray, index: usize) -> bool {
        array.validity().is_valid(index)
    }

    fn logical_validity(&self, array: &BoolArray) -> LogicalValidity {
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
    use crate::array::BoolArray;
    use crate::compute::scalar_at;
    use crate::validity::Validity;
    use crate::IntoArrayData;

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
}

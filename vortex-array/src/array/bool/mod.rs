use std::fmt::{Debug, Display};
use std::sync::Arc;

use arrow_array::BooleanArray;
use arrow_buffer::{BooleanBufferBuilder, MutableBuffer};
use itertools::Itertools;
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};

use crate::encoding::ids;
use crate::stats::StatsSet;
use crate::validity::{LogicalValidity, Validity, ValidityMetadata, ValidityVTable};
use crate::variants::{ArrayVariants, BoolArrayTrait};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{
    impl_encoding, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoArrayData, IntoCanonical,
};

pub mod compute;
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
    pub fn buffer(&self) -> &Buffer {
        self.as_ref()
            .buffer()
            .vortex_expect("Missing buffer in BoolArray")
    }

    /// Convert array into its internal buffer
    pub fn into_buffer(self) -> Buffer {
        self.into_array()
            .into_buffer()
            .vortex_expect("BoolArray must have a buffer")
    }

    /// Get array values as an arrow [BooleanBuffer]
    pub fn boolean_buffer(&self) -> BooleanBuffer {
        BooleanBuffer::new(
            self.buffer().clone().into_arrow(),
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
        let arrow_buffer = self.into_buffer().into_arrow();
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
            Some(Buffer::from(inner)),
            validity.into_array().into_iter().collect_vec().into(),
            StatsSet::default(),
        )?
        .try_into()
    }

    pub fn patch<P: AsPrimitive<usize>>(
        self,
        positions: &[P],
        values: BoolArray,
    ) -> VortexResult<Self> {
        if positions.len() != values.len() {
            vortex_bail!(
                "Positions and values passed to patch had different lengths {} and {}",
                positions.len(),
                values.len()
            );
        }
        if let Some(last_pos) = positions.last() {
            if last_pos.as_() >= self.len() {
                vortex_bail!(OutOfBounds: last_pos.as_(), 0, self.len())
            }
        }

        let len = self.len();
        let result_validity = self.validity().patch(len, positions, values.validity())?;
        let (mut own_values, bit_offset) = self.into_boolean_builder();
        for (idx, value) in positions.iter().zip_eq(values.boolean_buffer().iter()) {
            own_values.set_bit(idx.as_() + bit_offset, value);
        }

        Self::try_new(own_values.finish().slice(bit_offset, len), result_validity)
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

impl ArrayVariants for BoolArray {
    fn as_bool_array(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }
}

impl BoolArrayTrait for BoolArray {
    fn invert(&self) -> VortexResult<ArrayData> {
        Ok(BoolArray::try_new(!&self.boolean_buffer(), self.validity())?.into_array())
    }
}

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
    use arrow_buffer::BooleanBuffer;

    use crate::array::BoolArray;
    use crate::compute::slice;
    use crate::compute::unary::scalar_at;
    use crate::validity::Validity;
    use crate::{IntoArrayData, IntoArrayVariant};

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
        let arr = BoolArray::from(BooleanBuffer::new_set(12));
        let sliced = slice(arr, 4, 12).unwrap();
        let (values, offset) = sliced.into_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.as_slice(), &[255, 15]);
    }

    #[test]
    fn patch_sliced_bools_offset() {
        let arr = BoolArray::from(BooleanBuffer::new_set(15));
        let sliced = slice(arr, 4, 15).unwrap();
        let (values, offset) = sliced.into_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.as_slice(), &[255, 127]);
    }

    #[test]
    fn patch_sliced_bools_even() {
        let arr = BoolArray::from(BooleanBuffer::new_set(31));
        let sliced = slice(arr, 8, 24).unwrap();
        let (values, offset) = sliced.into_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 0);
        assert_eq!(values.as_slice(), &[255, 255]);
    }
}

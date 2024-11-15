use std::fmt::{Debug, Display};

use arrow_buffer::bit_iterator::{BitIndexIterator, BitSliceIterator};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, MutableBuffer};
use itertools::Itertools;
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};

use crate::array::visitor::{AcceptArrayVisitor, ArrayVisitor};
use crate::encoding::ids;
use crate::stats::StatsSet;
use crate::validity::{ArrayValidity, LogicalValidity, Validity, ValidityMetadata};
use crate::variants::{ArrayVariants, BoolArrayTrait};
use crate::{
    impl_encoding, ArrayData, ArrayTrait, Canonical, IntoArrayData, IntoCanonical, TypedArray,
};

mod accessors;
mod compute;
mod stats;

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

    pub fn try_new(buffer: BooleanBuffer, validity: Validity) -> VortexResult<Self> {
        let buffer_len = buffer.len();
        let buffer_offset = buffer.offset();
        let first_byte_bit_offset = (buffer_offset % 8) as u8;
        let buffer_byte_offset = buffer_offset - (first_byte_bit_offset as usize);

        let inner = buffer
            .into_inner()
            .bit_slice(buffer_byte_offset, buffer_len);

        Ok(Self {
            typed: TypedArray::try_from_parts(
                DType::Bool(validity.nullability()),
                buffer_len,
                BoolMetadata {
                    validity: validity.to_metadata(buffer_len)?,
                    first_byte_bit_offset,
                },
                Some(Buffer::from(inner)),
                validity.into_array().into_iter().collect_vec().into(),
                StatsSet::new(),
            )?,
        })
    }

    pub fn from_vec(bools: Vec<bool>, validity: Validity) -> Self {
        let buffer = BooleanBuffer::from(bools);
        Self::try_new(buffer, validity).vortex_expect("Failed to create BoolArray from vec")
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
}

impl ArrayTrait for BoolArray {}

impl ArrayVariants for BoolArray {
    fn as_bool_array(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }
}

impl BoolArrayTrait for BoolArray {
    fn invert(&self) -> VortexResult<ArrayData> {
        BoolArray::try_new(!&self.boolean_buffer(), self.validity()).map(|a| a.into_array())
    }

    fn maybe_null_indices_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> {
        Box::new(BitIndexIterator::new(self.buffer(), 0, self.len()))
    }

    fn maybe_null_slices_iter<'a>(&'a self) -> Box<dyn Iterator<Item = (usize, usize)> + 'a> {
        Box::new(BitSliceIterator::new(self.buffer(), 0, self.len()))
    }
}

impl From<BooleanBuffer> for BoolArray {
    fn from(value: BooleanBuffer) -> Self {
        Self::try_new(value, Validity::NonNullable)
            .vortex_expect("Failed to create BoolArray from BooleanBuffer")
    }
}

impl From<Vec<bool>> for BoolArray {
    fn from(value: Vec<bool>) -> Self {
        Self::from_vec(value, Validity::NonNullable)
    }
}

impl FromIterator<Option<bool>> for BoolArray {
    fn from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut validity: Vec<bool> = Vec::with_capacity(lower);
        let values: Vec<bool> = iter
            .map(|i| {
                validity.push(i.is_some());
                i.unwrap_or_default()
            })
            .collect::<Vec<_>>();

        Self::try_new(BooleanBuffer::from(values), Validity::from(validity))
            .vortex_expect("Failed to create BoolArray from iterator of Option<bool>")
    }
}

impl IntoCanonical for BoolArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        Ok(Canonical::Bool(self))
    }
}

impl ArrayValidity for BoolArray {
    fn is_valid(&self, index: usize) -> bool {
        self.validity().is_valid(index)
    }

    fn logical_validity(&self) -> LogicalValidity {
        self.validity().to_logical(self.len())
    }
}

impl AcceptArrayVisitor for BoolArray {
    fn accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_buffer(self.buffer())?;
        visitor.visit_validity(&self.validity())
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use itertools::Itertools;

    use crate::array::BoolArray;
    use crate::compute::slice;
    use crate::compute::unary::scalar_at;
    use crate::validity::Validity;
    use crate::variants::BoolArrayTrait;
    use crate::{IntoArrayData, IntoArrayVariant};

    #[test]
    fn bool_array() {
        let arr = BoolArray::from(vec![true, false, true]).into_array();
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
    fn constant_iter_true_test() {
        let arr = BoolArray::from(vec![true, true, true]);
        assert_eq!(vec![0, 1, 2], arr.maybe_null_indices_iter().collect_vec());
        assert_eq!(vec![(0, 3)], arr.maybe_null_slices_iter().collect_vec());
    }

    #[test]
    fn constant_iter_true_false_test() {
        let arr = BoolArray::from(vec![true, false, true]);
        assert_eq!(vec![0, 2], arr.maybe_null_indices_iter().collect_vec());
        assert_eq!(
            vec![(0, 1), (2, 3)],
            arr.maybe_null_slices_iter().collect_vec()
        );
    }

    #[test]
    fn constant_iter_false_test() {
        let arr = BoolArray::from(vec![false, false, false]);
        assert_eq!(0, arr.maybe_null_indices_iter().collect_vec().len());
        assert_eq!(0, arr.maybe_null_slices_iter().collect_vec().len());
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

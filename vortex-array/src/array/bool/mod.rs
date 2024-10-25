use std::fmt::{Debug, Display};

use arrow_buffer::bit_iterator::{BitIndexIterator, BitSliceIterator};
use arrow_buffer::BooleanBuffer;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult};

use crate::array::visitor::{AcceptArrayVisitor, ArrayVisitor};
use crate::encoding::ids;
use crate::stats::StatsSet;
use crate::validity::{ArrayValidity, LogicalValidity, Validity, ValidityMetadata};
use crate::variants::{ArrayVariants, BoolArrayTrait};
use crate::{impl_encoding, Array, ArrayTrait, Canonical, IntoArray, IntoCanonical, TypedArray};

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
    pub fn buffer(&self) -> &Buffer {
        self.as_ref()
            .buffer()
            .vortex_expect("Missing buffer in BoolArray")
    }

    pub fn boolean_buffer(&self) -> BooleanBuffer {
        BooleanBuffer::new(
            self.buffer().clone().into_arrow(),
            self.metadata().first_byte_bit_offset as usize,
            self.len(),
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
}

impl ArrayTrait for BoolArray {}

impl ArrayVariants for BoolArray {
    fn as_bool_array(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }
}

impl BoolArrayTrait for BoolArray {
    fn invert(&self) -> VortexResult<Array> {
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
    use itertools::Itertools;

    use crate::array::BoolArray;
    use crate::compute::unary::scalar_at;
    use crate::validity::Validity;
    use crate::variants::BoolArrayTrait;
    use crate::IntoArray;

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
}

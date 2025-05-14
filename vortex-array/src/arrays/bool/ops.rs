use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{BoolArray, BoolVTable};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{ArrayRef, Cost, IntoArray};

impl OperationsVTable<BoolVTable> for BoolVTable {
    fn slice(array: &BoolArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(BoolArray::new(
            array.boolean_buffer().slice(start, stop - start),
            array.validity().slice(start, stop)?,
        )
        .into_array())
    }

    fn scalar_at(array: &BoolArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::bool(
            array.boolean_buffer().value(index),
            array.dtype().nullability(),
        ))
    }

    fn is_constant(array: &BoolArray, cost: Cost) -> VortexResult<Option<bool>> {
        // If the array is small, then it is a constant time operation.
        if cost.is_negligible() && array.len() > 64 {
            return Ok(None);
        }

        let buffer = array.boolean_buffer();

        // Safety:
        // We must have at least one value at this point
        let first_value = unsafe { buffer.value_unchecked(0) };
        let value_block = if first_value { u64::MAX } else { 0_u64 };

        let bit_chunks = buffer.bit_chunks();
        let packed = bit_chunks.iter().all(|chunk| chunk == value_block);
        let reminder = bit_chunks.remainder_bits().count_ones() as usize
            == bit_chunks.remainder_len() * (first_value as usize);

        // We iterate on blocks of u64
        Ok(Some(packed & reminder))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToCanonical;
    use crate::compute::conformance::search_sorted::rstest;

    #[test]
    fn test_slice_large() {
        let arr = BoolArray::from_iter(std::iter::repeat_n(Some(true), 100));
        let sliced_arr = arr.slice(8, 16).unwrap().to_bool().unwrap();
        assert_eq!(sliced_arr.len(), 8);
        assert_eq!(sliced_arr.boolean_buffer().len(), 8);
        assert_eq!(sliced_arr.boolean_buffer().offset(), 0);
    }

    #[test]
    fn test_slice() {
        let arr = BoolArray::from_iter([Some(true), Some(true), None, Some(false), None]);
        let sliced_arr = arr.slice(1, 4).unwrap().to_bool().unwrap();

        assert_eq!(sliced_arr.len(), 3);

        let s = sliced_arr.scalar_at(0).unwrap();
        assert_eq!(s.as_bool().value(), Some(true));

        let s = sliced_arr.scalar_at(1).unwrap();
        assert!(!sliced_arr.is_valid(1).unwrap());
        assert!(s.is_null());

        let s = sliced_arr.scalar_at(2).unwrap();
        assert_eq!(s.as_bool().value(), Some(false));
    }

    #[rstest]
    #[case(vec![true], true)]
    #[case(vec![false; 65], true)]
    #[case({
        let mut v = vec![true; 64];
        v.push(false);
        v
    }, false)]
    fn test_is_constant(#[case] input: Vec<bool>, #[case] expected: bool) {
        let array = BoolArray::from_iter(input);
        assert_eq!(array.is_constant(), expected);
    }
}

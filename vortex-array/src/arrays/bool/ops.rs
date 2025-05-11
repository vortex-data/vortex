use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{BoolArray, BoolVTable};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToCanonical;

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
}

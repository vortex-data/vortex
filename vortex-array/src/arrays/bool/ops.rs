use vortex_error::VortexResult;

use crate::arrays::BoolArray;
use crate::{Array, ArrayOperationsImpl, ArrayRef};

impl ArrayOperationsImpl for BoolArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(BoolArray::new(
            self.boolean_buffer().slice(start, stop - start),
            self.validity().slice(start, stop)?,
        )
        .into_array())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToCanonical;
    use crate::compute::scalar_at;

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

        let s = scalar_at(&sliced_arr, 0).unwrap();
        assert_eq!(s.as_bool().value(), Some(true));

        let s = scalar_at(&sliced_arr, 1).unwrap();
        assert!(!sliced_arr.is_valid(1).unwrap());
        assert!(s.is_null());

        let s = scalar_at(&sliced_arr, 2).unwrap();
        assert_eq!(s.as_bool().value(), Some(false));
    }
}

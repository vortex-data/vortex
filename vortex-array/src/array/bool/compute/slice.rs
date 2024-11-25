use vortex_error::VortexResult;

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::SliceFn;
use crate::{ArrayData, IntoArrayData};

impl SliceFn<BoolArray> for BoolEncoding {
    fn slice(&self, array: &BoolArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(BoolArray::try_new(
            array.boolean_buffer().slice(start, stop - start),
            array.validity().slice(start, stop)?,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::slice;
    use crate::compute::unary::scalar_at;
    use crate::validity::ArrayValidity;
    use crate::ArrayLen;

    #[test]
    fn test_slice() {
        let arr = BoolArray::from_iter([Some(true), Some(true), None, Some(false), None]);
        let sliced_arr = slice(arr.as_ref(), 1, 4).unwrap();
        let sliced_arr = BoolArray::try_from(sliced_arr).unwrap();

        assert_eq!(sliced_arr.len(), 3);

        let s = scalar_at(sliced_arr.as_ref(), 0).unwrap();
        assert_eq!(s.as_bool().value(), Some(true));

        let s = scalar_at(sliced_arr.as_ref(), 1).unwrap();
        assert!(!sliced_arr.is_valid(1));
        assert!(s.is_null());

        let s = scalar_at(sliced_arr.as_ref(), 2).unwrap();
        assert_eq!(s.as_bool().value(), Some(false));
    }
}

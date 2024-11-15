use vortex_error::VortexResult;

use crate::array::BoolArray;
use crate::compute::SliceFn;
use crate::{ArrayData, IntoArrayData};

impl SliceFn for BoolArray {
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Self::try_new(
            self.boolean_buffer().slice(start, stop - start),
            self.validity().slice(start, stop)?,
        )
        .map(|a| a.into_array())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::compute::slice;
    use crate::compute::unary::scalar_at;
    use crate::validity::ArrayValidity;

    #[test]
    fn test_slice() {
        let arr = BoolArray::from_iter([Some(true), Some(true), None, Some(false), None]);
        let sliced_arr = slice(arr.as_ref(), 1, 4).unwrap();
        let sliced_arr = BoolArray::try_from(sliced_arr).unwrap();

        assert_eq!(sliced_arr.len(), 3);

        let s = scalar_at(sliced_arr.as_ref(), 0).unwrap();
        assert_eq!(s.into_value().as_bool().unwrap(), Some(true));

        let s = scalar_at(sliced_arr.as_ref(), 1).unwrap();
        assert!(!sliced_arr.is_valid(1));
        assert!(s.is_null());

        let s = scalar_at(sliced_arr.as_ref(), 2).unwrap();
        assert_eq!(s.into_value().as_bool().unwrap(), Some(false));
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Bool;
use crate::arrays::bool::BoolArrayExt;
use crate::scalar::Scalar;

impl OperationsVTable<Bool> for Bool {
    fn scalar_at(
        array: ArrayView<'_, Bool>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::bool(
            array.bit_buffer_view().value(index),
            array.dtype().nullability(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use crate::IntoArray;
    #[expect(deprecated)]
    use crate::ToCanonical as _;
    use crate::arrays::BoolArray;
    use crate::arrays::bool::BoolArrayExt;
    use crate::assert_arrays_eq;

    #[test]
    fn test_slice_hundred_elements() {
        let arr = BoolArray::from_iter(iter::repeat_n(Some(true), 100));
        #[expect(deprecated)]
        let sliced_arr = arr.into_array().slice(8..16).unwrap().to_bool();
        assert_eq!(sliced_arr.len(), 8);
        assert_eq!(sliced_arr.to_bit_buffer().len(), 8);
        assert_eq!(sliced_arr.to_bit_buffer().offset(), 0);
    }

    #[test]
    fn test_slice() {
        let arr = BoolArray::from_iter([Some(true), Some(true), None, Some(false), None]);
        #[expect(deprecated)]
        let sliced_arr = arr.into_array().slice(1..4).unwrap().to_bool();

        assert_arrays_eq!(
            sliced_arr,
            BoolArray::from_iter([Some(true), None, Some(false)])
        );
    }
}

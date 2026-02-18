// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::DictArray;
use crate::arrays::DictVTable;
use crate::arrays::SliceReduce;
use crate::scalar::Scalar;

impl SliceReduce for DictVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let sliced_code = array.codes().slice(range)?;
        // TODO(joe): if the range is size 1 replace with a constant array
        if let Some(code) = sliced_code.as_opt::<ConstantVTable>() {
            let code = code.scalar().as_primitive().as_::<usize>();
            return if let Some(code) = code {
                let values = array.values().slice(code..code + 1)?;
                Ok(Some(
                    DictArray::new(
                        ConstantArray::new(0u8, sliced_code.len()).into_array(),
                        values,
                    )
                    .into_array(),
                ))
            } else {
                Ok(Some(
                    ConstantArray::new(Scalar::null(array.dtype().clone()), sliced_code.len())
                        .to_array(),
                ))
            };
        }
        // SAFETY: slicing the codes preserves invariants.
        Ok(Some(
            unsafe { DictArray::new_unchecked(sliced_code, array.values().clone()) }.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::Nullable;
    use vortex_dtype::PType;
    use vortex_error::VortexResult;

    use crate::Array;
    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::scalar::Scalar;

    #[test]
    fn slice_constant_valid_code() -> VortexResult<()> {
        let dict = DictArray::new(
            ConstantArray::new(1u8, 5).into_array(),
            buffer![10i32, 20, 30].into_array(),
        );
        let sliced = dict.slice(1..4)?;
        let expected = PrimitiveArray::from_iter([20i32, 20, 20]).into_array();
        assert_arrays_eq!(sliced, expected);
        Ok(())
    }

    #[test]
    fn slice_constant_null_code() -> VortexResult<()> {
        let dict = DictArray::new(
            ConstantArray::new(Scalar::null(DType::Primitive(PType::U8, Nullable)), 5).into_array(),
            buffer![10i32, 20, 30].into_array(),
        );
        let sliced = dict.slice(1..4)?;
        let expected =
            PrimitiveArray::from_option_iter([Option::<i32>::None, None, None]).into_array();
        assert_arrays_eq!(sliced, expected);
        Ok(())
    }
}

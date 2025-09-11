// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::arrays::{ConstantArray, ConstantVTable};
use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::{DictArray, DictVTable};

impl OperationsVTable<DictVTable> for DictVTable {
    fn slice(array: &DictArray, range: Range<usize>) -> ArrayRef {
        let sliced_code = array.codes().slice(range);
        if sliced_code.is::<ConstantVTable>() {
            let code = &sliced_code.scalar_at(0).as_primitive().as_::<usize>();
            return if let Some(code) = code {
                ConstantArray::new(array.values().scalar_at(*code), sliced_code.len()).into_array()
            } else {
                ConstantArray::new(Scalar::null(array.dtype().clone()), sliced_code.len())
                    .to_array()
            };
        }
        // SAFETY: slicing the codes preserves invariants
        unsafe { DictArray::new_unchecked(sliced_code, array.values().clone()).into_array() }
    }

    fn scalar_at(array: &DictArray, index: usize) -> Scalar {
        let Some(dict_index) = array.codes().scalar_at(index).as_primitive().as_::<usize>() else {
            return Scalar::null(array.dtype().clone());
        };

        array
            .values()
            .scalar_at(dict_index)
            .cast(array.dtype())
            .vortex_expect("Array dtype will only differ by nullability")
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::DictArray;

    #[test]
    fn test_slice_into_const_dict() {
        let dict = DictArray::try_new(
            PrimitiveArray::from_option_iter(vec![Some(0u32), None, Some(1)]).to_array(),
            PrimitiveArray::from_option_iter(vec![Some(0i32), Some(1), Some(2)]).to_array(),
        )
        .unwrap();

        assert_eq!(
            Some(Scalar::new(dict.dtype().clone(), 0i32.into())),
            dict.slice(0..1).as_constant()
        );

        assert_eq!(
            Some(Scalar::null(dict.dtype().clone())),
            dict.slice(1..2).as_constant()
        );
    }

    #[test]
    fn test_scalar_at_null_code() {
        let dict = DictArray::try_new(
            PrimitiveArray::from_option_iter(vec![None, Some(0u32), None]).to_array(),
            buffer![1i32].into_array(),
        )
        .unwrap();

        assert_eq!(dict.scalar_at(0), Scalar::null(dict.dtype().clone()));
        assert_eq!(
            dict.scalar_at(1),
            Scalar::primitive(1, Nullability::Nullable)
        );
    }
}

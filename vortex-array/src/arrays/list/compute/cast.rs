// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{ListArray, ListVTable};
use crate::compute::{CastKernel, CastKernelAdapter, cast};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, register_kernel};

impl CastKernel for ListVTable {
    fn cast(&self, array: &Self::Array, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let Some(target_element_type) = dtype.as_list_element() else {
            return Ok(None);
        };

        let validity = array
            .validity()
            .clone()
            .cast_nullability(dtype.nullability())?;

        ListArray::try_new(
            cast(array.elements(), target_element_type)?,
            array.offsets().clone(),
            validity,
        )
        .map(|a| Some(a.to_array()))
    }
}

register_kernel!(CastKernelAdapter(ListVTable).lift());

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::{DType, Nullability};

    use crate::arrays::{BoolArray, ListArray, PrimitiveArray};
    use crate::compute::cast;
    use crate::validity::Validity;

    #[test]
    fn test_cast_list_success() {
        let list = ListArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4]).to_array(),
            PrimitiveArray::from_iter([0, 2, 3]).to_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let target_dtype = DType::List(
            Arc::new(DType::Primitive(
                vortex_dtype::PType::U64,
                Nullability::Nullable,
            )),
            Nullability::Nullable,
        );

        let result = cast(list.to_array().as_ref(), &target_dtype).unwrap();
        assert_eq!(result.dtype(), &target_dtype);
        assert_eq!(result.len(), list.len());
    }

    #[test]
    fn test_cast_to_wrong_type() {
        let list = ListArray::try_new(
            PrimitiveArray::from_iter([0i32, 2, 3, 4]).to_array(),
            PrimitiveArray::from_iter([0, 2, 3]).to_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let target_dtype = DType::Primitive(vortex_dtype::PType::U64, Nullability::NonNullable);
        // can't cast list to u64

        let result = cast(list.to_array().as_ref(), &target_dtype);
        assert!(result.is_err());
    }

    #[test]
    fn test_cant_cast_nulls_to_non_null() {
        // Test that if list has nulls, the conversion will fail

        // Nulls in the list itself
        let list = ListArray::try_new(
            PrimitiveArray::from_iter([0i32, 2, 3, 4]).to_array(),
            PrimitiveArray::from_iter([0, 2, 3]).to_array(),
            Validity::Array(BoolArray::from_iter(vec![false, true, true]).to_array()),
        )
        .unwrap();

        let target_dtype = DType::List(
            Arc::new(DType::Primitive(
                vortex_dtype::PType::U64,
                Nullability::Nullable,
            )),
            Nullability::NonNullable,
        );

        let result = cast(list.to_array().as_ref(), &target_dtype);
        assert!(result.is_err());

        // Nulls in list element array
        let list = ListArray::try_new(
            PrimitiveArray::from_option_iter([Some(0i32), Some(2), None, None]).to_array(),
            PrimitiveArray::from_iter([0, 2, 3]).to_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let target_dtype = DType::List(
            Arc::new(DType::Primitive(
                vortex_dtype::PType::U64,
                Nullability::NonNullable,
            )),
            Nullability::NonNullable,
        );

        let result = cast(list.to_array().as_ref(), &target_dtype);
        assert!(result.is_err());
    }
}

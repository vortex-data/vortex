use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::{ListArray, ListVTable};
use crate::compute::{CastKernel, CastKernelAdapter, cast};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, register_kernel};

impl CastKernel for ListVTable {
    fn cast(&self, array: &Self::Array, dtype: &DType) -> VortexResult<ArrayRef> {
        let Some(target_element_type) = dtype.as_list_element() else {
            vortex_bail!("cannot cast {} to {}", array.dtype(), dtype);
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
        .map(|a| a.to_array())
    }
}

register_kernel!(CastKernelAdapter(ListVTable).lift());

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::{DType, Nullability};

    use crate::arrays::{ListArray, PrimitiveArray};
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
    fn test_cast_list_fail() {
        let list = ListArray::try_new(
            PrimitiveArray::from_iter([0i32, 2, 3, 4]).to_array(),
            PrimitiveArray::from_iter([0, 2, 3]).to_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let target_dtype = DType::Primitive(vortex_dtype::PType::U64, Nullability::NonNullable);

        let result = cast(list.to_array().as_ref(), &target_dtype);
        assert!(result.is_err());
    }
}

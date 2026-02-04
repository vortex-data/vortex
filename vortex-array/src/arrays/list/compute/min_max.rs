// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::ListScalar;
use vortex_scalar::Scalar;

use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::register_kernel;

impl MinMaxKernel for ListVTable {
    fn min_max(&self, array: &ListArray) -> VortexResult<Option<MinMaxResult>> {
        let mut min: Option<Scalar> = None;
        let mut max: Option<Scalar> = None;
        for i in 0..array.len() {
            let scalar = array.scalar_at(i)?;
            if scalar.is_null() {
                continue;
            }
            let list_scalar = ListScalar::try_from(&scalar)?;
            if let Some(current_min) = &min {
                let current_min_list = ListScalar::try_from(current_min)?;
                if list_scalar < current_min_list {
                    min = Some(scalar.cast(&array.dtype().as_nonnullable())?);
                }
            } else {
                min = Some(scalar.cast(&array.dtype().as_nonnullable())?);
            }
            if let Some(current_max) = &max {
                let current_max_list = ListScalar::try_from(current_max)?;
                if list_scalar > current_max_list {
                    max = Some(scalar.cast(&array.dtype().as_nonnullable())?);
                }
            } else {
                max = Some(scalar.cast(&array.dtype().as_nonnullable())?);
            }
        }
        match (min, max) {
            (Some(min), Some(max)) => Ok(Some(MinMaxResult { min, max })),
            (None, None) => Ok(None),
            _ => unreachable!("min and max should be set together or both remain None"),
        }
    }
}

register_kernel!(MinMaxKernelAdapter(ListVTable).lift());

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::ArrayRef;
    use crate::builders::{ArrayBuilder as _, ListBuilder};
    use crate::compute::min_max;

    fn create_list_array(values: Vec<Option<Vec<i32>>>, validity: Vec<bool>) -> ArrayRef {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let nullability = Nullability::Nullable;
        let mut builder = ListBuilder::<u32>::new(element_dtype, nullability);
        for i in 0..validity.len() {
            if validity[i] {
                let value = values[i].as_ref().unwrap();
                let list_scalar = value.into_iter().map(|x| Scalar::primitive(*x, Nullability::NonNullable)).collect::<Vec<_>>();
                let list = Scalar::list(Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)), list_scalar, Nullability::NonNullable);
                builder.append_value(list.as_list()).unwrap();
            } else {
                builder.append_null();
            }
        }
        builder.finish()
    }

    #[test]
    fn test_list_min_max_i32() {
        // Lists: [1,2], [2,3], [3,4]
        let list = create_list_array(vec![Some(vec![1,2]), Some(vec![2,3]), Some(vec![3,4])], vec![true, true, true]);
        let result = min_max(list.as_ref()).unwrap().unwrap();
        assert_eq!(result.min, vec![1i32, 2].into());
        assert_eq!(result.max, vec![3i32, 4].into());
    }

    #[test]
    fn test_list_min_max_with_nulls() {
        // Lists: null, [1,2], [2,3]
        let list = create_list_array(vec![None, Some(vec![1,2]), Some(vec![2,3])], vec![false, true, true]);
        let result = min_max(list.as_ref()).unwrap().unwrap();
        assert_eq!(result.min, vec![1i32, 2].into());
        assert_eq!(result.max, vec![2i32, 3].into());
    }

    #[test]
    fn test_list_min_max_all_nulls() {
        // All nulls
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let nullability = Nullability::Nullable;
        let mut builder = ListBuilder::<u32>::new(element_dtype, nullability);
        builder.append_null();
        let list = builder.finish();
        assert!(min_max(list.as_ref()).unwrap().is_none());
    }

    #[test]
    fn test_list_min_max_empty() {
        // Empty array
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let nullability = Nullability::Nullable;
        let mut builder = ListBuilder::<u32>::new(element_dtype, nullability);
        let list = builder.finish();
        assert!(min_max(list.as_ref()).unwrap().is_none());
    }
}

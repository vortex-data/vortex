// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::DictArray;
use super::DictVTable;
use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::compute::CompareKernel;
use crate::compute::CompareKernelAdapter;
use crate::compute::Operator;
use crate::compute::compare;
use crate::register_kernel;

impl CompareKernel for DictVTable {
    fn compare(
        &self,
        lhs: &DictArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        // if we have more values than codes, it is faster to canonicalise first.
        if lhs.values().len() > lhs.codes().len() {
            return Ok(None);
        }

        // If the RHS is constant, then we just need to compare against our encoded values.
        if let Some(rhs) = rhs.as_constant() {
            let compare_result = compare(
                lhs.values(),
                ConstantArray::new(rhs, lhs.values().len()).as_ref(),
                operator,
            )?;

            // SAFETY: values len preserved, codes all still point to valid values
            let result = unsafe {
                DictArray::new_unchecked(lhs.codes().clone(), compare_result)
                    .set_all_values_referenced(lhs.has_all_values_referenced())
                    .into_array()
            };

            // We canonicalize the result because dictionary-encoded bools is dumb.
            return Ok(Some(result.to_canonical()?.into_array()));
        }

        // It's a little more complex, but we could perform a comparison against the dictionary
        // values in the future.
        Ok(None)
    }
}

register_kernel!(CompareKernelAdapter(DictVTable).lift());
#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::dict::DictArray;
    use crate::assert_arrays_eq;
    use crate::compute::Operator;
    use crate::compute::compare;
    use crate::validity::Validity;

    #[test]
    fn test_compare_value() {
        let dict = DictArray::try_new(
            buffer![0u32, 1, 2].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap();

        let res = compare(
            dict.as_ref(),
            ConstantArray::new(Scalar::from(1i32), 3).as_ref(),
            Operator::Eq,
        )
        .unwrap();
        assert_arrays_eq!(res, BoolArray::from_iter([true, false, false]));
    }

    #[test]
    fn test_compare_non_eq() {
        let dict = DictArray::try_new(
            buffer![0u32, 1, 2].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap();

        let res = compare(
            dict.as_ref(),
            ConstantArray::new(Scalar::from(1i32), 3).as_ref(),
            Operator::Gt,
        )
        .unwrap();
        assert_arrays_eq!(res, BoolArray::from_iter([false, true, true]));
    }

    #[test]
    fn test_compare_nullable() {
        let dict = DictArray::try_new(
            PrimitiveArray::new(
                buffer![0u32, 1, 2],
                Validity::from_iter([false, true, false]),
            )
            .into_array(),
            PrimitiveArray::new(buffer![1i32, 2, 3], Validity::AllValid).into_array(),
        )
        .unwrap();

        let res = compare(
            dict.as_ref(),
            ConstantArray::new(Scalar::primitive(4i32, Nullability::Nullable), 3).as_ref(),
            Operator::Eq,
        )
        .unwrap();
        assert_arrays_eq!(res, BoolArray::from_iter([None, Some(false), None]));
        assert_eq!(res.dtype().nullability(), Nullability::Nullable);
        assert_eq!(
            res.validity_mask().unwrap(),
            Mask::from_iter([false, true, false])
        );
    }

    #[test]
    fn test_compare_null_values() {
        let dict = DictArray::try_new(
            buffer![0u32, 1, 2].into_array(),
            PrimitiveArray::new(
                buffer![1i32, 2, 0],
                Validity::from_iter([true, true, false]),
            )
            .into_array(),
        )
        .unwrap();

        let res = compare(
            dict.as_ref(),
            ConstantArray::new(Scalar::primitive(4i32, Nullability::NonNullable), 3).as_ref(),
            Operator::Eq,
        )
        .unwrap();
        assert_arrays_eq!(res, BoolArray::from_iter([Some(false), Some(false), None]));
        assert_eq!(res.dtype().nullability(), Nullability::Nullable);
        assert_eq!(
            res.validity_mask().unwrap(),
            Mask::from_iter([true, true, false])
        );
    }
}

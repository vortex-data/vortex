// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::canonical::ToCanonical;
use crate::compute::CompareKernel;
use crate::compute::CompareKernelAdapter;
use crate::compute::Operator;
use crate::compute::compare;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

impl CompareKernel for MaskedVTable {
    fn compare(
        &self,
        lhs: &MaskedArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        // Compare the child arrays
        let compare_result = compare(&lhs.child, rhs, operator)?;

        // Get the boolean buffer from the comparison result
        let bool_array = compare_result.to_bool();
        let combined_validity = bool_array.validity().clone().and(lhs.validity().clone());

        // Return a plain BoolArray with the combined validity
        Ok(Some(
            BoolArray::new(bool_array.to_bit_buffer(), combined_validity).into_array(),
        ))
    }
}

register_kernel!(CompareKernelAdapter(MaskedVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability;
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::compute::Operator;
    use crate::compute::compare;
    use crate::validity::Validity;

    #[test]
    fn test_compare_value() {
        let masked = MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            Validity::AllValid,
        )
        .unwrap();

        let res = compare(
            masked.as_ref(),
            ConstantArray::new(Scalar::from(2i32), 3).as_ref(),
            Operator::Eq,
        )
        .unwrap();
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([Some(false), Some(true), Some(false)])
        );
    }

    #[test]
    fn test_compare_non_eq() {
        let masked = MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            Validity::AllValid,
        )
        .unwrap();

        let res = compare(
            masked.as_ref(),
            ConstantArray::new(Scalar::from(2i32), 3).as_ref(),
            Operator::Gt,
        )
        .unwrap();
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([Some(false), Some(false), Some(true)])
        );
    }

    #[test]
    fn test_compare_nullable() {
        // MaskedArray with nulls
        let masked = MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            Validity::from_iter([false, true, false]),
        )
        .unwrap();

        let res = compare(
            masked.as_ref(),
            ConstantArray::new(Scalar::primitive(2i32, Nullability::Nullable), 3).as_ref(),
            Operator::Eq,
        )
        .unwrap();
        assert_arrays_eq!(res, BoolArray::from_iter([None, Some(true), None]));
        assert_eq!(res.dtype().nullability(), Nullability::Nullable);
        assert_eq!(
            res.validity_mask().unwrap(),
            Mask::from_iter([false, true, false])
        );
    }

    #[test]
    fn test_compare_with_null_rhs() {
        // MaskedArray with some nulls
        let masked = MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            Validity::from_iter([true, true, false]),
        )
        .unwrap();

        // RHS has a null value
        let rhs = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]);

        let res = compare(masked.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();
        assert_arrays_eq!(res, BoolArray::from_iter([Some(true), None, None]));
        assert_eq!(res.dtype().nullability(), Nullability::Nullable);
        // Validity is union of both: lhs=[T,T,F], rhs=[T,F,T] => result=[T,F,F]
        assert_eq!(
            res.validity_mask().unwrap(),
            Mask::from_iter([true, false, false])
        );
    }
}

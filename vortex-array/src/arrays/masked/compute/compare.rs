// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{BoolArray, MaskedArray, MaskedVTable};
use crate::canonical::ToCanonical;
use crate::compute::{CompareKernel, CompareKernelAdapter, Operator, compare};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, register_kernel};

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
            BoolArray::from_bool_buffer(bool_array.boolean_buffer().clone(), combined_validity)
                .into_array(),
        ))
    }
}

register_kernel!(CompareKernelAdapter(MaskedVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability;
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::arrays::{ConstantArray, MaskedArray, PrimitiveArray};
    use crate::compute::{Operator, compare};
    use crate::validity::Validity;
    use crate::{IntoArray, ToCanonical};

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
        let res = res.to_bool();
        assert_eq!(
            res.boolean_buffer().iter().collect::<Vec<_>>(),
            vec![false, true, false]
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
        let res = res.to_bool();
        assert_eq!(
            res.boolean_buffer().iter().collect::<Vec<_>>(),
            vec![false, false, true]
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
        let res = res.to_bool();
        assert_eq!(
            res.boolean_buffer().iter().collect::<Vec<_>>(),
            vec![false, true, false]
        );
        assert_eq!(res.dtype().nullability(), Nullability::Nullable);
        assert_eq!(res.validity_mask(), Mask::from_iter([false, true, false]));
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
        let res = res.to_bool();
        assert_eq!(
            res.boolean_buffer().iter().collect::<Vec<_>>(),
            vec![true, false, true]
        );
        assert_eq!(res.dtype().nullability(), Nullability::Nullable);
        // Validity is union of both: lhs=[T,T,F], rhs=[T,F,T] => result=[T,F,F]
        assert_eq!(res.validity_mask(), Mask::from_iter([true, false, false]));
    }
}

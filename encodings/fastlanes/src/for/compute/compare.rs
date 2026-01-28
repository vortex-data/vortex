// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Shr;

use num_traits::WrappingSub;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::CompareKernel;
use vortex_array::compute::CompareKernelAdapter;
use vortex_array::compute::Operator;
use vortex_array::compute::compare;
use vortex_array::register_kernel;
use vortex_dtype::NativePType;
use vortex_dtype::Nullability;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexError;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_scalar::PValue;
use vortex_scalar::PrimitiveScalar;
use vortex_scalar::Scalar;

use crate::FoRArray;
use crate::FoRVTable;

impl CompareKernel for FoRVTable {
    fn compare(
        &self,
        lhs: &FoRArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(constant) = rhs.as_constant()
            && let Ok(constant) = PrimitiveScalar::try_from(&constant)
        {
            match_each_integer_ptype!(constant.ptype(), |T| {
                return compare_constant(
                    lhs,
                    constant
                        .typed_value::<T>()
                        .vortex_expect("null scalar handled in top-level"),
                    rhs.dtype().nullability(),
                    operator,
                );
            })
        }

        Ok(None)
    }
}

register_kernel!(CompareKernelAdapter(FoRVTable).lift());

fn compare_constant<T>(
    lhs: &FoRArray,
    mut rhs: T,
    nullability: Nullability,
    operator: Operator,
) -> VortexResult<Option<ArrayRef>>
where
    T: NativePType + WrappingSub + Shr<usize, Output = T>,
    T: TryFrom<PValue, Error = VortexError>,
    PValue: From<T>,
{
    // For now, we only support equals and not equals. Comparisons are a little more fiddly to
    // get right regarding how to handle overflow and the wrapping subtraction.
    if !matches!(operator, Operator::Eq | Operator::NotEq) {
        return Ok(None);
    }

    let reference = lhs.reference_scalar();
    let reference = reference.as_primitive().typed_value::<T>();

    // We encode the RHS into the FoR domain.
    if let Some(reference) = reference {
        rhs = rhs.wrapping_sub(&reference);
    }

    // Wrap up the RHS into a scalar and cast to the encoded DType (this will be the equivalent
    // unsigned integer type).
    let rhs = Scalar::primitive(rhs, nullability);

    compare(
        lhs.encoded(),
        ConstantArray::new(rhs, lhs.len()).as_ref(),
        operator,
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;

    use super::*;

    #[test]
    fn test_compare_constant() {
        let reference = Scalar::from(10);
        // 10, 30, 12
        let lhs = FoRArray::try_new(
            PrimitiveArray::new(buffer!(0i32, 20, 2), Validity::AllValid).into_array(),
            reference,
        )
        .unwrap();

        let result = compare_constant(&lhs, 30i32, Nullability::NonNullable, Operator::Eq)
            .unwrap()
            .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([false, true, false].map(Some)));

        let result = compare_constant(&lhs, 12i32, Nullability::NonNullable, Operator::NotEq)
            .unwrap()
            .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([true, true, false].map(Some)));

        for op in [Operator::Lt, Operator::Lte, Operator::Gt, Operator::Gte] {
            assert!(
                compare_constant(&lhs, 30i32, Nullability::NonNullable, op)
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn test_compare_nullable_constant() {
        let reference = Scalar::from(0);
        // 10, 30, 12
        let lhs = FoRArray::try_new(
            PrimitiveArray::new(buffer!(0i32, 20, 2), Validity::NonNullable).into_array(),
            reference,
        )
        .unwrap();

        assert_eq!(
            compare_constant(&lhs, 30i32, Nullability::Nullable, Operator::Eq)
                .unwrap()
                .unwrap()
                .dtype(),
            &DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            compare_constant(&lhs, 30i32, Nullability::NonNullable, Operator::Eq)
                .unwrap()
                .unwrap()
                .dtype(),
            &DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn compare_non_encodable_constant() {
        let reference = Scalar::from(10);
        // 10, 30, 12
        let lhs = FoRArray::try_new(
            PrimitiveArray::new(buffer!(0i32, 10, 1), Validity::AllValid).into_array(),
            reference,
        )
        .unwrap();

        let result = compare_constant(&lhs, -1i32, Nullability::NonNullable, Operator::Eq)
            .unwrap()
            .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, false, false].map(Some))
        );

        let result = compare_constant(&lhs, -1i32, Nullability::NonNullable, Operator::NotEq)
            .unwrap()
            .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([true, true, true].map(Some)));
    }

    #[test]
    fn compare_large_constant() {
        let reference = Scalar::from(-9219218377546224477i64);
        #[allow(clippy::cast_possible_truncation)]
        let lhs = FoRArray::try_new(
            PrimitiveArray::new(
                buffer![0i64, 9654309310445864926u64 as i64],
                Validity::AllValid,
            )
            .into_array(),
            reference,
        )
        .unwrap();

        let result = compare_constant(
            &lhs,
            435090932899640449i64,
            Nullability::Nullable,
            Operator::Eq,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([Some(false), Some(true)]));

        let result = compare_constant(
            &lhs,
            435090932899640449i64,
            Nullability::Nullable,
            Operator::NotEq,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([Some(true), Some(false)]));
    }
}

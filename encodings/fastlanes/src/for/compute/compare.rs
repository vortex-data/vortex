// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Shr;

use num_traits::WrappingSub;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexError;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;

use crate::FoR;
use crate::r#for::array::FoRArrayExt;

impl CompareKernel for FoR {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(constant) = rhs.as_constant()
            && let Some(constant) = constant.as_primitive_opt()
        {
            match_each_integer_ptype!(constant.ptype(), |T| {
                return compare_constant(
                    lhs,
                    constant
                        .typed_value::<T>()
                        .vortex_expect("null scalar handled in adaptor"),
                    rhs.dtype().nullability(),
                    operator,
                );
            })
        }

        Ok(None)
    }
}

fn compare_constant<T>(
    lhs: ArrayView<'_, FoR>,
    mut rhs: T,
    nullability: Nullability,
    operator: CompareOperator,
) -> VortexResult<Option<ArrayRef>>
where
    T: NativePType + WrappingSub + Shr<usize, Output = T>,
    T: TryFrom<PValue, Error = VortexError>,
    PValue: From<T>,
{
    // For now, we only support equals and not equals. Comparisons are a little more fiddly to
    // get right regarding how to handle overflow and the wrapping subtraction.
    if !matches!(operator, CompareOperator::Eq | CompareOperator::NotEq) {
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

    lhs.encoded()
        .binary(
            ConstantArray::new(rhs, lhs.len()).into_array(),
            Operator::from(operator),
        )
        .map(Some)
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::*;
    use crate::FoR;
    use crate::FoRArray;

    fn for_arr(encoded: ArrayRef, reference: Scalar) -> FoRArray {
        FoR::try_new(encoded, reference).vortex_expect("FoR array construction should succeed")
    }

    #[test]
    fn test_compare_constant() {
        let reference = Scalar::from(10);
        // 10, 30, 12
        let lhs = for_arr(
            PrimitiveArray::new(buffer!(0i32, 20, 2), Validity::AllValid).into_array(),
            reference,
        );

        let result = compare_constant(
            lhs.as_view(),
            30i32,
            Nullability::NonNullable,
            CompareOperator::Eq,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([false, true, false].map(Some)));

        let result = compare_constant(
            lhs.as_view(),
            12i32,
            Nullability::NonNullable,
            CompareOperator::NotEq,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([true, true, false].map(Some)));

        for op in [
            CompareOperator::Lt,
            CompareOperator::Lte,
            CompareOperator::Gt,
            CompareOperator::Gte,
        ] {
            assert!(
                compare_constant(lhs.as_view(), 30i32, Nullability::NonNullable, op)
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn test_compare_nullable_constant() {
        let reference = Scalar::from(0);
        // 10, 30, 12
        let lhs = for_arr(
            PrimitiveArray::new(buffer!(0i32, 20, 2), Validity::NonNullable).into_array(),
            reference,
        );

        assert_eq!(
            compare_constant(
                lhs.as_view(),
                30i32,
                Nullability::Nullable,
                CompareOperator::Eq,
            )
            .unwrap()
            .unwrap()
            .dtype(),
            &DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            compare_constant(
                lhs.as_view(),
                30i32,
                Nullability::NonNullable,
                CompareOperator::Eq,
            )
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
        let lhs = for_arr(
            PrimitiveArray::new(buffer!(0i32, 10, 1), Validity::AllValid).into_array(),
            reference,
        );

        let result = compare_constant(
            lhs.as_view(),
            -1i32,
            Nullability::NonNullable,
            CompareOperator::Eq,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, false, false].map(Some))
        );

        let result = compare_constant(
            lhs.as_view(),
            -1i32,
            Nullability::NonNullable,
            CompareOperator::NotEq,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([true, true, true].map(Some)));
    }

    #[test]
    fn compare_large_constant() {
        let reference = Scalar::from(-9219218377546224477i64);
        let lhs = for_arr(
            PrimitiveArray::new(
                buffer![0i64, 9654309310445864926u64 as i64],
                Validity::AllValid,
            )
            .into_array(),
            reference,
        );

        let result = compare_constant(
            lhs.as_view(),
            435090932899640449i64,
            Nullability::Nullable,
            CompareOperator::Eq,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([Some(false), Some(true)]));

        let result = compare_constant(
            lhs.as_view(),
            435090932899640449i64,
            Nullability::Nullable,
            CompareOperator::NotEq,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([Some(true), Some(false)]));
    }
}

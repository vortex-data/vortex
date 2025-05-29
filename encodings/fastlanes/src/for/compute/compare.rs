use std::ops::Shr;

use num_traits::WrappingSub;
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{CompareKernel, CompareKernelAdapter, Operator, compare};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_dtype::{NativePType, Nullability, match_each_integer_ptype};
use vortex_error::{VortexError, VortexExpect as _, VortexResult};
use vortex_scalar::{PValue, PrimitiveScalar, Scalar};

use crate::{FoRArray, FoRVTable};

impl CompareKernel for FoRVTable {
    fn compare(
        &self,
        lhs: &FoRArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(constant) = rhs.as_constant() {
            if let Ok(constant) = PrimitiveScalar::try_from(&constant) {
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
    let rhs = Scalar::primitive(rhs, nullability).reinterpret_cast(T::PTYPE.to_unsigned());

    compare(
        lhs.encoded(),
        ConstantArray::new(rhs, lhs.len()).as_ref(),
        operator,
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::DType;

    use super::*;

    #[test]
    fn test_compare_constant() {
        let reference = Scalar::from(10);
        // 10, 30, 12
        let lhs = FoRArray::try_new(
            PrimitiveArray::new(buffer!(0u32, 20, 2), Validity::AllValid).into_array(),
            reference,
        )
        .unwrap();

        assert_result(
            compare_constant(&lhs, 30i32, Nullability::NonNullable, Operator::Eq),
            [false, true, false],
        );
        assert_result(
            compare_constant(&lhs, 12i32, Nullability::NonNullable, Operator::NotEq),
            [true, true, false],
        );
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
            PrimitiveArray::new(buffer!(0u32, 20, 2), Validity::NonNullable).into_array(),
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
            PrimitiveArray::new(buffer!(0u32, 10, 1), Validity::AllValid).into_array(),
            reference,
        )
        .unwrap();

        assert_result(
            compare_constant(&lhs, -1i32, Nullability::NonNullable, Operator::Eq),
            [false, false, false],
        );
        assert_result(
            compare_constant(&lhs, -1i32, Nullability::NonNullable, Operator::NotEq),
            [true, true, true],
        );
    }

    #[test]
    fn compare_large_constant() {
        let reference = Scalar::from(-9219218377546224477i64);
        let lhs = FoRArray::try_new(
            PrimitiveArray::new(buffer![0u64, 9654309310445864926], Validity::AllValid)
                .into_array(),
            reference,
        )
        .unwrap();

        assert_result(
            compare_constant(
                &lhs,
                435090932899640449i64,
                Nullability::NonNullable,
                Operator::Eq,
            ),
            [false, true],
        );
        assert_result(
            compare_constant(
                &lhs,
                435090932899640449i64,
                Nullability::NonNullable,
                Operator::NotEq,
            ),
            [true, false],
        );
    }

    fn assert_result<T: IntoIterator<Item = bool>>(
        result: VortexResult<Option<ArrayRef>>,
        expected: T,
    ) {
        let result = result.unwrap().unwrap().to_bool().unwrap();
        assert_eq!(result.boolean_buffer(), &BooleanBuffer::from_iter(expected));
    }
}

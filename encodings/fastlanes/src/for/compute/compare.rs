// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Shr;

use num_traits::WrappingSub;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
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
        // Handle comparison with constant
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

        // Handle FoR vs FoR comparison
        if let Some(rhs_for) = rhs.as_opt::<FoRVTable>() {
            // If both have the same reference, we can compare encoded values directly
            if lhs.reference_scalar() == rhs_for.reference_scalar() {
                return compare(lhs.encoded(), rhs_for.encoded(), operator).map(Some);
            }
            // Otherwise, fall through to canonical comparison
        }

        Ok(None)
    }
}

register_kernel!(CompareKernelAdapter(FoRVTable).lift());

fn compare_constant<T>(
    lhs: &FoRArray,
    rhs: T,
    nullability: Nullability,
    operator: Operator,
) -> VortexResult<Option<ArrayRef>>
where
    T: NativePType + WrappingSub + Shr<usize, Output = T> + PartialOrd,
    T: TryFrom<PValue, Error = VortexError>,
    PValue: From<T>,
{
    let reference_scalar = lhs.reference_scalar();
    let reference = reference_scalar.as_primitive().typed_value::<T>();

    // For FoR encoding, reference is the minimum value and encoded = value - reference.
    // This means all encoded values are >= 0.
    //
    // For inequality comparisons, we can use this property:
    // - value op constant <=> encoded op (constant - reference)
    //
    // But we need to handle edge cases where constant < reference:
    // - For Lt/Lte: if constant < reference, all values >= reference > constant, result is all false
    // - For Gt/Gte: if constant < reference, all values >= reference > constant, result is all true
    match operator {
        Operator::Lt | Operator::Lte => {
            if let Some(ref_val) = reference
                && rhs < ref_val
            {
                // constant < reference (min), all values > constant
                return Ok(Some(
                    ConstantArray::new(Scalar::bool(false, nullability), lhs.len()).into_array(),
                ));
            }
            // Transform constant and compare in encoded domain
            let transformed_rhs = if let Some(ref_val) = reference {
                rhs.wrapping_sub(&ref_val)
            } else {
                rhs
            };
            let rhs_scalar = Scalar::primitive(transformed_rhs, nullability);
            compare(
                lhs.encoded(),
                ConstantArray::new(rhs_scalar, lhs.len()).as_ref(),
                operator,
            )
            .map(Some)
        }
        Operator::Gt | Operator::Gte => {
            if let Some(ref_val) = reference
                && rhs < ref_val
            {
                // constant < reference (min), all values >= reference > constant
                return Ok(Some(
                    ConstantArray::new(Scalar::bool(true, nullability), lhs.len()).into_array(),
                ));
            }
            // Transform constant and compare in encoded domain
            let transformed_rhs = if let Some(ref_val) = reference {
                rhs.wrapping_sub(&ref_val)
            } else {
                rhs
            };
            let rhs_scalar = Scalar::primitive(transformed_rhs, nullability);
            compare(
                lhs.encoded(),
                ConstantArray::new(rhs_scalar, lhs.len()).as_ref(),
                operator,
            )
            .map(Some)
        }
        Operator::Eq | Operator::NotEq => {
            // Original logic for equality - wrapping subtraction is safe
            let transformed_rhs = if let Some(ref_val) = reference {
                rhs.wrapping_sub(&ref_val)
            } else {
                rhs
            };
            let rhs_scalar = Scalar::primitive(transformed_rhs, nullability);
            compare(
                lhs.encoded(),
                ConstantArray::new(rhs_scalar, lhs.len()).as_ref(),
                operator,
            )
            .map(Some)
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;

    use super::*;

    #[test]
    fn test_compare_constant() {
        let reference = Scalar::from(10);
        // Values: 10, 30, 12 (encoded as 0, 20, 2)
        let lhs = FoRArray::try_new(
            PrimitiveArray::new(buffer!(0i32, 20, 2), Validity::AllValid).into_array(),
            reference,
        )
        .unwrap();

        // Equality tests
        assert_result(
            compare_constant(&lhs, 30i32, Nullability::NonNullable, Operator::Eq),
            [false, true, false],
        );
        assert_result(
            compare_constant(&lhs, 12i32, Nullability::NonNullable, Operator::NotEq),
            [true, true, false],
        );

        // Inequality tests: values are [10, 30, 12]
        // Less than 15: [10 < 15, 30 < 15, 12 < 15] = [true, false, true]
        assert_result(
            compare_constant(&lhs, 15i32, Nullability::NonNullable, Operator::Lt),
            [true, false, true],
        );
        // Less than or equal to 12: [10 <= 12, 30 <= 12, 12 <= 12] = [true, false, true]
        assert_result(
            compare_constant(&lhs, 12i32, Nullability::NonNullable, Operator::Lte),
            [true, false, true],
        );
        // Greater than 12: [10 > 12, 30 > 12, 12 > 12] = [false, true, false]
        assert_result(
            compare_constant(&lhs, 12i32, Nullability::NonNullable, Operator::Gt),
            [false, true, false],
        );
        // Greater than or equal to 12: [10 >= 12, 30 >= 12, 12 >= 12] = [false, true, true]
        assert_result(
            compare_constant(&lhs, 12i32, Nullability::NonNullable, Operator::Gte),
            [false, true, true],
        );
    }

    #[test]
    fn test_compare_constant_below_reference() {
        let reference = Scalar::from(10);
        // Values: 10, 30, 12 (reference/min is 10)
        let lhs = FoRArray::try_new(
            PrimitiveArray::new(buffer!(0i32, 20, 2), Validity::AllValid).into_array(),
            reference,
        )
        .unwrap();

        // When constant < reference (min), all values are > constant
        // Lt 5: all values >= 10 > 5, so none are < 5 => all false
        assert_result(
            compare_constant(&lhs, 5i32, Nullability::NonNullable, Operator::Lt),
            [false, false, false],
        );
        // Lte 5: all values >= 10 > 5, so none are <= 5 => all false
        assert_result(
            compare_constant(&lhs, 5i32, Nullability::NonNullable, Operator::Lte),
            [false, false, false],
        );
        // Gt 5: all values >= 10 > 5, so all are > 5 => all true
        assert_result(
            compare_constant(&lhs, 5i32, Nullability::NonNullable, Operator::Gt),
            [true, true, true],
        );
        // Gte 5: all values >= 10 > 5, so all are >= 5 => all true
        assert_result(
            compare_constant(&lhs, 5i32, Nullability::NonNullable, Operator::Gte),
            [true, true, true],
        );
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

        assert_result(
            compare_constant(
                &lhs,
                435090932899640449i64,
                Nullability::Nullable,
                Operator::Eq,
            ),
            [false, true],
        );
        assert_result(
            compare_constant(
                &lhs,
                435090932899640449i64,
                Nullability::Nullable,
                Operator::NotEq,
            ),
            [true, false],
        );
    }

    fn assert_result<T: IntoIterator<Item = bool>>(
        result: VortexResult<Option<ArrayRef>>,
        expected: T,
    ) {
        let result = result.unwrap().unwrap().to_bool();
        assert_eq!(result.bit_buffer(), &BitBuffer::from_iter(expected));
    }
}

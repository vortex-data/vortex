// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare implementations for DecimalScalar enum.

use vortex_dtype::NativeDecimalType;
use vortex_dtype::i256;
use vortex_error::vortex_panic;
use vortex_vector::bool::BoolScalar;
use vortex_vector::decimal::DScalar;
use vortex_vector::decimal::DecimalScalar;
use vortex_vector::match_each_dscalar_pair;

use crate::comparison::Compare;
use crate::comparison::ComparisonOperator;

impl<Op, D> Compare<Op> for DScalar<D>
where
    D: NativeDecimalType,
    Op: ComparisonOperator<D>,
{
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        match (self.value(), rhs.value()) {
            (Some(l), Some(r)) => BoolScalar::new(Some(Op::apply(&l, &r))),
            _ => BoolScalar::new(None),
        }
    }
}

impl<Op> Compare<Op> for DecimalScalar
where
    DScalar<i8>: Compare<Op, Output = BoolScalar>,
    DScalar<i16>: Compare<Op, Output = BoolScalar>,
    DScalar<i32>: Compare<Op, Output = BoolScalar>,
    DScalar<i64>: Compare<Op, Output = BoolScalar>,
    DScalar<i128>: Compare<Op, Output = BoolScalar>,
    DScalar<i256>: Compare<Op, Output = BoolScalar>,
{
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        match_each_dscalar_pair!((self, rhs), |l, r| { Compare::<Op>::compare(l, r) }, {
            vortex_panic!("Cannot compare DecimalScalars of different types")
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::PrecisionScale;

    use super::*;
    use crate::comparison::Equal;
    use crate::comparison::GreaterThan;
    use crate::comparison::LessThan;
    use crate::comparison::NotEqual;

    #[test]
    fn test_dscalar_compare_i32() {
        let ps = PrecisionScale::<i32>::new(9, 2);
        let left = unsafe { DScalar::new_unchecked(ps, Some(5i32)) };
        let right = unsafe { DScalar::new_unchecked(ps, Some(3i32)) };

        assert_eq!(
            Compare::<Equal>::compare(left.clone(), right.clone()).value(),
            Some(false)
        );
        assert_eq!(
            Compare::<NotEqual>::compare(left.clone(), right.clone()).value(),
            Some(true)
        );
        assert_eq!(
            Compare::<GreaterThan>::compare(left.clone(), right.clone()).value(),
            Some(true)
        );
        assert_eq!(
            Compare::<LessThan>::compare(left, right).value(),
            Some(false)
        );
    }

    #[test]
    fn test_decimal_scalar_compare() {
        let ps = PrecisionScale::<i64>::new(18, 4);
        let left: DecimalScalar = unsafe { DScalar::new_unchecked(ps, Some(10i64)) }.into();
        let right: DecimalScalar = unsafe { DScalar::new_unchecked(ps, Some(10i64)) }.into();

        assert_eq!(
            Compare::<Equal>::compare(left.clone(), right.clone()).value(),
            Some(true)
        );
        assert_eq!(
            Compare::<NotEqual>::compare(left, right).value(),
            Some(false)
        );
    }

    #[test]
    fn test_dscalar_compare_with_null() {
        let ps = PrecisionScale::<i32>::new(9, 2);
        let left = unsafe { DScalar::new_unchecked(ps, Some(5i32)) };
        let right = unsafe { DScalar::new_unchecked(ps, None) };

        // Comparison with null returns null
        assert_eq!(Compare::<Equal>::compare(left, right).value(), None);
    }
}

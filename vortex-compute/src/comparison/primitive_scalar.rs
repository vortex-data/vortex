// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare implementations for PrimitiveScalar enum.

use vortex_dtype::NativePType;
use vortex_dtype::half::f16;
use vortex_error::vortex_panic;
use vortex_vector::bool::BoolScalar;
use vortex_vector::match_each_pscalar_pair;
use vortex_vector::primitive::PScalar;
use vortex_vector::primitive::PrimitiveScalar;

use crate::comparison::Compare;
use crate::comparison::ComparisonOperator;

impl<Op, T> Compare<Op> for PScalar<T>
where
    T: NativePType,
    Op: ComparisonOperator<T>,
{
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        match (self.value(), rhs.value()) {
            (Some(l), Some(r)) => BoolScalar::new(Some(Op::apply(&l, &r))),
            _ => vortex_panic!("cannot compare primitive scalar with different types"),
        }
    }
}

impl<Op> Compare<Op> for PrimitiveScalar
where
    PScalar<i8>: Compare<Op, Output = BoolScalar>,
    PScalar<i16>: Compare<Op, Output = BoolScalar>,
    PScalar<i32>: Compare<Op, Output = BoolScalar>,
    PScalar<i64>: Compare<Op, Output = BoolScalar>,
    PScalar<u8>: Compare<Op, Output = BoolScalar>,
    PScalar<u16>: Compare<Op, Output = BoolScalar>,
    PScalar<u32>: Compare<Op, Output = BoolScalar>,
    PScalar<u64>: Compare<Op, Output = BoolScalar>,
    PScalar<f16>: Compare<Op, Output = BoolScalar>,
    PScalar<f32>: Compare<Op, Output = BoolScalar>,
    PScalar<f64>: Compare<Op, Output = BoolScalar>,
{
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        match_each_pscalar_pair!((self, rhs), |l, r| { Compare::<Op>::compare(l, r) }, {
            vortex_panic!("Cannot compare PrimitiveScalars of different types",)
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_vector::primitive::PScalar;

    use super::*;
    use crate::comparison::Equal;
    use crate::comparison::GreaterThan;
    use crate::comparison::LessThan;
    use crate::comparison::NotEqual;

    #[test]
    fn test_pscalar_compare_i32() {
        let left = PScalar::new(Some(5i32));
        let right = PScalar::new(Some(3i32));

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
    fn test_primitive_scalar_compare() {
        let left: PrimitiveScalar = PScalar::new(Some(10u64)).into();
        let right: PrimitiveScalar = PScalar::new(Some(10u64)).into();

        assert_eq!(
            Compare::<Equal>::compare(left.clone(), right.clone()).value(),
            Some(true)
        );
        assert_eq!(
            Compare::<NotEqual>::compare(left, right).value(),
            Some(false)
        );
    }
}

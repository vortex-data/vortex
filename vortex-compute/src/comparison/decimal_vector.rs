// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare implementations for DecimalVector enum.

use vortex_dtype::i256;
use vortex_error::vortex_panic;
use vortex_vector::bool::BoolVector;
use vortex_vector::decimal::DVector;
use vortex_vector::decimal::DecimalVector;
use vortex_vector::match_each_dvector_pair;

use crate::comparison::Compare;

impl<Op> Compare<Op> for &DecimalVector
where
    for<'a> &'a DVector<i8>: Compare<Op, Output = BoolVector>,
    for<'a> &'a DVector<i16>: Compare<Op, Output = BoolVector>,
    for<'a> &'a DVector<i32>: Compare<Op, Output = BoolVector>,
    for<'a> &'a DVector<i64>: Compare<Op, Output = BoolVector>,
    for<'a> &'a DVector<i128>: Compare<Op, Output = BoolVector>,
    for<'a> &'a DVector<i256>: Compare<Op, Output = BoolVector>,
{
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        match_each_dvector_pair!((self, rhs), |l, r| { Compare::<Op>::compare(l, r) }, {
            vortex_panic!(
                "Cannot compare DecimalVectors of different types: {:?} and {:?}",
                self,
                rhs
            )
        })
    }
}

impl<Op> Compare<Op> for DecimalVector
where
    for<'a> &'a DecimalVector: Compare<Op, Output = BoolVector>,
{
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        Compare::<Op>::compare(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::PrecisionScale;
    use vortex_mask::Mask;
    use vortex_vector::VectorOps;
    use vortex_vector::decimal::DVector;

    use super::*;
    use crate::comparison::Equal;
    use crate::comparison::LessThan;

    #[test]
    fn test_compare_i32() {
        let ps = PrecisionScale::<i32>::new(9, 2);
        let left: DecimalVector = DVector::new(ps, buffer![1i32, 2, 3], Mask::new_true(3)).into();
        let right: DecimalVector = DVector::new(ps, buffer![1i32, 3, 2], Mask::new_true(3)).into();

        let result = Compare::<Equal>::compare(&left, &right);
        assert_eq!(result.validity(), &Mask::new_true(3));
        // 1==1, 2!=3, 3!=2
        assert_eq!(result.scalar_at(0).value(), Some(true));
        assert_eq!(result.scalar_at(1).value(), Some(false));
        assert_eq!(result.scalar_at(2).value(), Some(false));
    }

    #[test]
    fn test_compare_i64() {
        let ps = PrecisionScale::<i64>::new(18, 4);
        let left: DecimalVector = DVector::new(ps, buffer![1i64, 2, 3], Mask::new_true(3)).into();
        let right: DecimalVector = DVector::new(ps, buffer![0i64, 2, 4], Mask::new_true(3)).into();

        let result = Compare::<LessThan>::compare(&left, &right);
        // 1 < 0? false, 2 < 2? false, 3 < 4? true
        assert_eq!(result.scalar_at(0).value(), Some(false));
        assert_eq!(result.scalar_at(1).value(), Some(false));
        assert_eq!(result.scalar_at(2).value(), Some(true));
    }
}

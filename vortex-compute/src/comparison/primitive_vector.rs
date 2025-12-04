// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare implementations for PrimitiveVector enum.

use vortex_dtype::half::f16;
use vortex_error::vortex_panic;
use vortex_vector::bool::BoolVector;
use vortex_vector::match_each_pvector_pair;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PrimitiveVector;

use crate::comparison::Compare;

impl<Op> Compare<Op> for &PrimitiveVector
where
    for<'a> &'a PVector<i8>: Compare<Op, Output = BoolVector>,
    for<'a> &'a PVector<i16>: Compare<Op, Output = BoolVector>,
    for<'a> &'a PVector<i32>: Compare<Op, Output = BoolVector>,
    for<'a> &'a PVector<i64>: Compare<Op, Output = BoolVector>,
    for<'a> &'a PVector<u8>: Compare<Op, Output = BoolVector>,
    for<'a> &'a PVector<u16>: Compare<Op, Output = BoolVector>,
    for<'a> &'a PVector<u32>: Compare<Op, Output = BoolVector>,
    for<'a> &'a PVector<u64>: Compare<Op, Output = BoolVector>,
    for<'a> &'a PVector<f16>: Compare<Op, Output = BoolVector>,
    for<'a> &'a PVector<f32>: Compare<Op, Output = BoolVector>,
    for<'a> &'a PVector<f64>: Compare<Op, Output = BoolVector>,
{
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        match_each_pvector_pair!((self, rhs), |l, r| { Compare::<Op>::compare(l, r) }, {
            vortex_panic!(
                "Cannot compare PrimitiveVectors of different types: {:?} and {:?}",
                self,
                rhs
            )
        })
    }
}

impl<Op> Compare<Op> for PrimitiveVector
where
    for<'a> &'a PrimitiveVector: Compare<Op, Output = BoolVector>,
{
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        Compare::<Op>::compare(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use vortex_mask::Mask;
    use vortex_vector::VectorMutOps;
    use vortex_vector::VectorOps;
    use vortex_vector::primitive::PVectorMut;

    use super::*;
    use crate::comparison::Equal;
    use crate::comparison::LessThan;

    #[test]
    fn test_compare_i32() {
        let left: PrimitiveVector = PVectorMut::from_iter([1i32, 2, 3].map(Some))
            .freeze()
            .into();
        let right: PrimitiveVector = PVectorMut::from_iter([1i32, 3, 2].map(Some))
            .freeze()
            .into();

        let result = Compare::<Equal>::compare(&left, &right);
        assert_eq!(result.validity(), &Mask::new_true(3));
        // 1==1, 2!=3, 3!=2
        assert_eq!(result.scalar_at(0).value(), Some(true));
        assert_eq!(result.scalar_at(1).value(), Some(false));
        assert_eq!(result.scalar_at(2).value(), Some(false));
    }

    #[test]
    fn test_compare_f64() {
        let left: PrimitiveVector = PVectorMut::from_iter([1.0f64, 2.0, 3.0].map(Some))
            .freeze()
            .into();
        let right: PrimitiveVector = PVectorMut::from_iter([0.0f64, 2.0, 4.0].map(Some))
            .freeze()
            .into();

        let result = Compare::<LessThan>::compare(&left, &right);
        // 1.0 < 0.0? false, 2.0 < 2.0? false, 3.0 < 4.0? true
        assert_eq!(result.scalar_at(0).value(), Some(false));
        assert_eq!(result.scalar_at(1).value(), Some(false));
        assert_eq!(result.scalar_at(2).value(), Some(true));
    }
}

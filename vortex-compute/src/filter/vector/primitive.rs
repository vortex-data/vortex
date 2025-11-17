// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::half::f16;
use vortex_vector::primitive::{PVector, PVectorMut, PrimitiveVector, PrimitiveVectorMut};
use vortex_vector::{match_each_pvector, match_each_pvector_mut};

use crate::filter::Filter;

impl<M> Filter<M> for &PrimitiveVector
where
    for<'a> &'a PVector<i8>: Filter<M, Output = PVector<i8>>,
    for<'a> &'a PVector<i16>: Filter<M, Output = PVector<i16>>,
    for<'a> &'a PVector<i32>: Filter<M, Output = PVector<i32>>,
    for<'a> &'a PVector<i64>: Filter<M, Output = PVector<i64>>,
    for<'a> &'a PVector<u8>: Filter<M, Output = PVector<u8>>,
    for<'a> &'a PVector<u16>: Filter<M, Output = PVector<u16>>,
    for<'a> &'a PVector<u32>: Filter<M, Output = PVector<u32>>,
    for<'a> &'a PVector<u64>: Filter<M, Output = PVector<u64>>,
    for<'a> &'a PVector<f16>: Filter<M, Output = PVector<f16>>,
    for<'a> &'a PVector<f32>: Filter<M, Output = PVector<f32>>,
    for<'a> &'a PVector<f64>: Filter<M, Output = PVector<f64>>,
{
    type Output = PrimitiveVector;

    fn filter(self, selection: &M) -> Self::Output {
        match_each_pvector!(self, |v| { v.filter(selection).into() })
    }
}

impl<M> Filter<M> for &mut PrimitiveVectorMut
where
    for<'a> &'a mut PVectorMut<i8>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<i16>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<i32>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<i64>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<u8>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<u16>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<u32>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<u64>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<f16>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<f32>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<f64>: Filter<M, Output = ()>,
{
    type Output = ();

    fn filter(self, selection: &M) -> Self::Output {
        match_each_pvector_mut!(self, |v| { v.filter(selection) })
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::PTypeDowncast;
    use vortex_mask::Mask;
    use vortex_vector::primitive::PVectorMut;
    use vortex_vector::{VectorMutOps, VectorOps};

    use super::*;
    use crate::filter::MaskIndices;

    #[test]
    fn test_filter_primitive_vector_with_mask() {
        let vec = PrimitiveVector::from(
            PVectorMut::<i32>::from_iter([100, 200, 300, 400, 500].map(Some)).freeze(),
        );

        let mask = Mask::from_iter([true, false, true, false, true]);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 3);
        let p32 = filtered.into_i32();
        assert_eq!(p32.get(0), Some(&100));
        assert_eq!(p32.get(1), Some(&300));
        assert_eq!(p32.get(2), Some(&500));
    }

    #[test]
    fn test_filter_primitive_vector_with_mask_indices() {
        let vec = PrimitiveVector::from(
            PVectorMut::<i32>::from_iter([100, 200, 300, 400, 500].map(Some)).freeze(),
        );

        let indices = unsafe { MaskIndices::new_unchecked(&[0, 2, 4]) };

        let filtered = vec.filter(&indices);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 3);
        let p32 = filtered.into_i32();
        assert_eq!(p32.get(0), Some(&100));
        assert_eq!(p32.get(1), Some(&300));
        assert_eq!(p32.get(2), Some(&500));
    }

    #[test]
    fn test_filter_primitive_vector_with_nulls() {
        let vec = PrimitiveVector::from(
            PVectorMut::<i64>::from_iter([Some(1000), None, Some(3000), Some(4000), None]).freeze(),
        );

        let mask = Mask::from_iter([true, true, false, true, false]);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 2);
        let p64 = filtered.into_i64();
        assert_eq!(p64.get(0), Some(&1000));
        assert_eq!(p64.get(1), None);
        assert_eq!(p64.get(2), Some(&4000));
    }

    #[test]
    fn test_filter_primitive_vector_all_true() {
        let vec =
            PrimitiveVector::from(PVectorMut::<i32>::from_iter([100, 200, 300].map(Some)).freeze());

        let mask = Mask::new_true(3);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        let p32 = filtered.into_i32();
        assert_eq!(p32.get(0), Some(&100));
        assert_eq!(p32.get(1), Some(&200));
        assert_eq!(p32.get(2), Some(&300));
    }

    #[test]
    fn test_filter_primitive_vector_all_false() {
        let vec =
            PrimitiveVector::from(PVectorMut::<i32>::from_iter([100, 200, 300].map(Some)).freeze());

        let mask = Mask::new_false(3);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_filter_primitive_vector_mut_with_mask() {
        let mut vec = PrimitiveVectorMut::from(PVectorMut::<i32>::from_iter(
            [100, 200, 300, 400, 500].map(Some),
        ));

        let mask = Mask::from_iter([true, false, true, false, true]);

        vec.filter(&mask);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 3);
        let p32 = frozen.into_i32();
        assert_eq!(p32.get(0), Some(&100));
        assert_eq!(p32.get(1), Some(&300));
        assert_eq!(p32.get(2), Some(&500));
    }

    #[test]
    fn test_filter_primitive_vector_mut_with_mask_indices() {
        let mut vec = PrimitiveVectorMut::from(PVectorMut::<i32>::from_iter(
            [100, 200, 300, 400, 500].map(Some),
        ));

        let indices = unsafe { MaskIndices::new_unchecked(&[0, 2, 4]) };

        vec.filter(&indices);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 3);
        let p32 = frozen.into_i32();
        assert_eq!(p32.get(0), Some(&100));
        assert_eq!(p32.get(1), Some(&300));
        assert_eq!(p32.get(2), Some(&500));
    }

    #[test]
    fn test_filter_primitive_vector_mut_with_nulls() {
        let mut vec = PrimitiveVectorMut::from(PVectorMut::<i64>::from_iter([
            Some(1000),
            None,
            Some(3000),
            Some(4000),
            None,
        ]));

        let mask = Mask::from_iter([true, true, false, true, false]);

        vec.filter(&mask);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 2);
        let p64 = frozen.into_i64();
        assert_eq!(p64.get(0), Some(&1000));
        assert_eq!(p64.get(1), None);
        assert_eq!(p64.get(2), Some(&4000));
    }
}

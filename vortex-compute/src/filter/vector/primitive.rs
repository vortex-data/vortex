// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;
use vortex_vector::primitive::{PrimitiveVector, PrimitiveVectorMut};
use vortex_vector::{match_each_pvector, match_each_pvector_mut};

use crate::filter::{Filter, MaskIndices};

impl Filter<Mask> for &PrimitiveVector {
    type Output = PrimitiveVector;

    fn filter(self, selection_mask: &Mask) -> PrimitiveVector {
        match_each_pvector!(self, |v| { v.filter(selection_mask).into() })
    }
}

impl Filter<MaskIndices<'_>> for &PrimitiveVector {
    type Output = PrimitiveVector;

    fn filter(self, indices: &MaskIndices<'_>) -> Self::Output {
        match_each_pvector!(self, |v| { v.filter(indices).into() })
    }
}

impl Filter<Mask> for &mut PrimitiveVectorMut {
    type Output = ();

    fn filter(self, selection_mask: &Mask) {
        match_each_pvector_mut!(self, |v| { v.filter(selection_mask) })
    }
}

impl Filter<MaskIndices<'_>> for &mut PrimitiveVectorMut {
    type Output = ();

    fn filter(self, indices: &MaskIndices<'_>) -> Self::Output {
        match_each_pvector_mut!(self, |v| { v.filter(indices) })
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::PTypeDowncast;
    use vortex_mask::Mask;
    use vortex_vector::primitive::PVectorMut;
    use vortex_vector::{VectorMutOps, VectorOps};

    use super::*;

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

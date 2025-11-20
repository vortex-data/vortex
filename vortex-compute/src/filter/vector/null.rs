// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitView;
use vortex_mask::Mask;
use vortex_vector::null::{NullVector, NullVectorMut};

use crate::filter::Filter;

impl Filter<Mask> for &NullVector {
    type Output = NullVector;

    fn filter(self, selection: &Mask) -> Self::Output {
        NullVector::new(selection.true_count())
    }
}

impl<const NB: usize> Filter<BitView<'_, NB>> for &NullVector {
    type Output = NullVector;

    fn filter(self, selection: &BitView<'_, NB>) -> Self::Output {
        NullVector::new(selection.true_count())
    }
}

impl Filter<Mask> for &mut NullVectorMut {
    type Output = ();

    fn filter(self, selection: &Mask) -> Self::Output {
        *self = NullVectorMut::new(selection.true_count())
    }
}

impl<const NB: usize> Filter<BitView<'_, NB>> for &mut NullVectorMut {
    type Output = ();

    fn filter(self, selection: &BitView<'_, NB>) -> Self::Output {
        *self = NullVectorMut::new(selection.true_count())
    }
}

#[cfg(test)]
mod tests {
    use vortex_mask::Mask;
    use vortex_vector::{VectorMutOps, VectorOps};

    use super::*;

    #[test]
    fn test_filter_null_vector_with_mask() {
        let vec = NullVector::new(5);
        let mask = Mask::from_iter([true, false, true, false, true]);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 0);
    }

    #[test]
    fn test_filter_null_vector_all_true() {
        let vec = NullVector::new(3);
        let mask = Mask::new_true(3);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 0);
    }

    #[test]
    fn test_filter_null_vector_all_false() {
        let vec = NullVector::new(3);
        let mask = Mask::new_false(3);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_filter_null_vector_mut_with_mask() {
        let mut vec = NullVectorMut::new(5);
        let mask = Mask::from_iter([true, false, true, false, true]);

        vec.filter(&mask);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 0);
    }

    #[test]
    fn test_filter_null_vector_mut_all_true() {
        let mut vec = NullVectorMut::new(3);
        let mask = Mask::new_true(3);

        vec.filter(&mask);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 0);
    }

    #[test]
    fn test_filter_null_vector_mut_all_false() {
        let mut vec = NullVectorMut::new(3);
        let mask = Mask::new_false(3);

        vec.filter(&mask);

        assert_eq!(vec.len(), 0);
    }
}

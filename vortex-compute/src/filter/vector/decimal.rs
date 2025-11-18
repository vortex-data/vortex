// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::i256;
use vortex_vector::decimal::{DVector, DVectorMut, DecimalVector, DecimalVectorMut};
use vortex_vector::{match_each_dvector, match_each_dvector_mut};

use crate::filter::Filter;

impl<M> Filter<M> for &DecimalVector
where
    for<'a> &'a DVector<i8>: Filter<M, Output = DVector<i8>>,
    for<'a> &'a DVector<i16>: Filter<M, Output = DVector<i16>>,
    for<'a> &'a DVector<i32>: Filter<M, Output = DVector<i32>>,
    for<'a> &'a DVector<i64>: Filter<M, Output = DVector<i64>>,
    for<'a> &'a DVector<i128>: Filter<M, Output = DVector<i128>>,
    for<'a> &'a DVector<i256>: Filter<M, Output = DVector<i256>>,
{
    type Output = DecimalVector;

    fn filter(self, selection: &M) -> Self::Output {
        match_each_dvector!(self, |d| { d.filter(selection).into() })
    }
}

impl<M> Filter<M> for &mut DecimalVectorMut
where
    for<'a> &'a mut DVectorMut<i8>: Filter<M, Output = ()>,
    for<'a> &'a mut DVectorMut<i16>: Filter<M, Output = ()>,
    for<'a> &'a mut DVectorMut<i32>: Filter<M, Output = ()>,
    for<'a> &'a mut DVectorMut<i64>: Filter<M, Output = ()>,
    for<'a> &'a mut DVectorMut<i128>: Filter<M, Output = ()>,
    for<'a> &'a mut DVectorMut<i256>: Filter<M, Output = ()>,
{
    type Output = ();

    fn filter(self, selection: &M) -> Self::Output {
        match_each_dvector_mut!(self, |d| { d.filter(selection) });
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BufferMut;
    use vortex_dtype::{DecimalTypeDowncast, PrecisionScale};
    use vortex_mask::{Mask, MaskMut};
    use vortex_vector::decimal::DVectorMut;
    use vortex_vector::{VectorMutOps, VectorOps};

    use super::*;
    use crate::filter::MaskIndices;

    #[test]
    fn test_filter_decimal_vector_with_mask() {
        let ps = PrecisionScale::<i32>::new(9, 2);
        let elements = BufferMut::from_iter([100_i32, 200, 300, 400, 500]);
        let validity = MaskMut::new_true(5);
        let vec = DecimalVector::from(DVectorMut::new(ps, elements, validity).freeze());

        let mask = Mask::from_iter([true, false, true, false, true]);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 3);
        let d32 = filtered.into_i32();
        assert_eq!(d32.get(0), Some(&100));
        assert_eq!(d32.get(1), Some(&300));
        assert_eq!(d32.get(2), Some(&500));
    }

    #[test]
    fn test_filter_decimal_vector_with_mask_indices() {
        let ps = PrecisionScale::<i32>::new(9, 2);
        let elements = BufferMut::from_iter([100_i32, 200, 300, 400, 500]);
        let validity = MaskMut::new_true(5);
        let vec = DecimalVector::from(DVectorMut::new(ps, elements, validity).freeze());

        let indices = unsafe { MaskIndices::new_unchecked(&[0, 2, 4]) };

        let filtered = vec.filter(&indices);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 3);
        let d32 = filtered.into_i32();
        assert_eq!(d32.get(0), Some(&100));
        assert_eq!(d32.get(1), Some(&300));
        assert_eq!(d32.get(2), Some(&500));
    }

    #[test]
    fn test_filter_decimal_vector_with_nulls() {
        let ps = PrecisionScale::<i64>::new(18, 4);
        let elements = BufferMut::from_iter([1000_i64, 0, 3000, 4000, 0]);
        let mut validity = MaskMut::with_capacity(5);
        validity.append_n(true, 1);
        validity.append_n(false, 1);
        validity.append_n(true, 1);
        validity.append_n(true, 1);
        validity.append_n(false, 1);
        let vec = DecimalVector::from(DVectorMut::new(ps, elements, validity).freeze());

        let mask = Mask::from_iter([true, true, false, true, false]);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 2);
        let d64 = filtered.into_i64();
        assert_eq!(d64.get(0), Some(&1000));
        assert_eq!(d64.get(1), None);
        assert_eq!(d64.get(2), Some(&4000));
    }

    #[test]
    fn test_filter_decimal_vector_all_true() {
        let ps = PrecisionScale::<i32>::new(9, 2);
        let elements = BufferMut::from_iter([100_i32, 200, 300]);
        let validity = MaskMut::new_true(3);
        let vec = DecimalVector::from(DVectorMut::new(ps, elements, validity).freeze());

        let mask = Mask::new_true(3);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        let d32 = filtered.into_i32();
        assert_eq!(d32.get(0), Some(&100));
        assert_eq!(d32.get(1), Some(&200));
        assert_eq!(d32.get(2), Some(&300));
    }

    #[test]
    fn test_filter_decimal_vector_all_false() {
        let ps = PrecisionScale::<i32>::new(9, 2);
        let elements = BufferMut::from_iter([100_i32, 200, 300]);
        let validity = MaskMut::new_true(3);
        let vec = DVectorMut::new(ps, elements, validity).freeze();

        let mask = Mask::new_false(3);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_filter_decimal_vector_mut_with_mask() {
        let ps = PrecisionScale::<i32>::new(9, 2);
        let elements = BufferMut::from_iter([100_i32, 200, 300, 400, 500]);
        let validity = MaskMut::new_true(5);
        let mut vec = DecimalVectorMut::from(DVectorMut::new(ps, elements, validity));

        let mask = Mask::from_iter([true, false, true, false, true]);

        vec.filter(&mask);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 3);
        let d32 = frozen.into_i32();
        assert_eq!(d32.get(0), Some(&100));
        assert_eq!(d32.get(1), Some(&300));
        assert_eq!(d32.get(2), Some(&500));
    }

    #[test]
    fn test_filter_decimal_vector_mut_with_mask_indices() {
        let ps = PrecisionScale::<i32>::new(9, 2);
        let elements = BufferMut::from_iter([100_i32, 200, 300, 400, 500]);
        let validity = MaskMut::new_true(5);
        let mut vec = DecimalVectorMut::from(DVectorMut::new(ps, elements, validity));

        let indices = unsafe { MaskIndices::new_unchecked(&[0, 2, 4]) };

        vec.filter(&indices);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 3);
        let d32 = frozen.into_i32();
        assert_eq!(d32.get(0), Some(&100));
        assert_eq!(d32.get(1), Some(&300));
        assert_eq!(d32.get(2), Some(&500));
    }

    #[test]
    fn test_filter_decimal_vector_mut_with_nulls() {
        let ps = PrecisionScale::<i64>::new(18, 4);
        let elements = BufferMut::from_iter([1000_i64, 0, 3000, 4000, 0]);
        let mut validity = MaskMut::with_capacity(5);
        validity.append_n(true, 1);
        validity.append_n(false, 1);
        validity.append_n(true, 1);
        validity.append_n(true, 1);
        validity.append_n(false, 1);
        let mut vec = DecimalVectorMut::from(DVectorMut::new(ps, elements, validity));

        let mask = Mask::from_iter([true, true, false, true, false]);

        vec.filter(&mask);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 2);
        let d64 = frozen.into_i64();
        assert_eq!(d64.get(0), Some(&1000));
        assert_eq!(d64.get(1), None);
        assert_eq!(d64.get(2), Some(&4000));
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::compute::{MaskKernel, MaskKernelAdapter, TakeKernel, TakeKernelAdapter};
use vortex_array::vtable::ValidityHelper;
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::{ByteBoolArray, ByteBoolVTable};

impl MaskKernel for ByteBoolVTable {
    fn mask(&self, array: &ByteBoolArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(ByteBoolArray::new(array.buffer().clone(), array.validity().mask(mask)?).into_array())
    }
}

register_kernel!(MaskKernelAdapter(ByteBoolVTable).lift());

impl TakeKernel for ByteBoolVTable {
    fn take(&self, array: &ByteBoolArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = indices.to_primitive()?;
        let bools = array.as_slice();

        // This handles combining validity from both source array and nullable indices
        let validity = array.validity().take(indices.as_ref())?;

        let taken_bools = match_each_integer_ptype!(indices.ptype(), |I| {
            indices
                .as_slice::<I>()
                .iter()
                .map(|&idx| {
                    let idx: usize = idx.as_();
                    bools[idx]
                })
                .collect::<Vec<bool>>()
        });

        Ok(ByteBoolArray::from_vec(taken_bools, validity).into_array())
    }
}

register_kernel!(TakeKernelAdapter(ByteBoolVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::compute::{Operator, compare};

    use super::*;

    #[test]
    fn test_slice() {
        let original = vec![Some(true), Some(true), None, Some(false), None];
        let vortex_arr = ByteBoolArray::from(original);

        let sliced_arr = vortex_arr.slice(1, 4).unwrap();
        let sliced_arr = sliced_arr.as_::<ByteBoolVTable>();

        let s = sliced_arr.scalar_at(0).unwrap();
        assert_eq!(s.as_bool().value(), Some(true));

        let s = sliced_arr.scalar_at(1).unwrap();
        assert!(!sliced_arr.is_valid(1).unwrap());
        assert!(s.is_null());
        assert_eq!(s.as_bool().value(), None);

        let s = sliced_arr.scalar_at(2).unwrap();
        assert_eq!(s.as_bool().value(), Some(false));
    }

    #[test]
    fn test_compare_all_equal() {
        let lhs = ByteBoolArray::from(vec![true; 5]);
        let rhs = ByteBoolArray::from(vec![true; 5]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        for i in 0..arr.len() {
            let s = arr.scalar_at(i).unwrap();
            assert!(s.is_valid());
            assert_eq!(s.as_bool().value(), Some(true));
        }
    }

    #[test]
    fn test_compare_all_different() {
        let lhs = ByteBoolArray::from(vec![false; 5]);
        let rhs = ByteBoolArray::from(vec![true; 5]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        for i in 0..arr.len() {
            let s = arr.scalar_at(i).unwrap();
            assert!(s.is_valid());
            assert_eq!(s.as_bool().value(), Some(false));
        }
    }

    #[test]
    fn test_compare_with_nulls() {
        let lhs = ByteBoolArray::from(vec![true; 5]);
        let rhs = ByteBoolArray::from(vec![Some(true), Some(true), Some(true), Some(false), None]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        for i in 0..3 {
            let s = arr.scalar_at(i).unwrap();
            assert!(s.is_valid());
            assert_eq!(s.as_bool().value(), Some(true));
        }

        let s = arr.scalar_at(3).unwrap();
        assert!(s.is_valid());
        assert_eq!(s.as_bool().value(), Some(false));

        let s = arr.scalar_at(4).unwrap();
        assert!(s.is_null());
    }

    #[test]
    fn test_mask_byte_bool() {
        test_mask_conformance(ByteBoolArray::from(vec![true, false, true, true, false]).as_ref());
        test_mask_conformance(
            ByteBoolArray::from(vec![Some(true), Some(true), None, Some(false), None]).as_ref(),
        );
    }

    #[test]
    fn test_filter_byte_bool() {
        test_filter_conformance(ByteBoolArray::from(vec![true, false, true, true, false]).as_ref());
        test_filter_conformance(
            ByteBoolArray::from(vec![Some(true), Some(true), None, Some(false), None]).as_ref(),
        );
    }

    #[rstest]
    #[case(ByteBoolArray::from(vec![true, false, true, true, false]))]
    #[case(ByteBoolArray::from(vec![Some(true), Some(true), None, Some(false), None]))]
    #[case(ByteBoolArray::from(vec![true, false]))]
    #[case(ByteBoolArray::from(vec![true]))]
    fn test_take_byte_bool_conformance(#[case] array: ByteBoolArray) {
        test_take_conformance(array.as_ref());
    }
}

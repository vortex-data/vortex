use num_traits::AsPrimitive;
use vortex_array::compute::{MaskKernel, MaskKernelAdapter, TakeKernel, TakeKernelAdapter};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayRef, ToCanonical, register_kernel};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::{ByteBoolArray, ByteBoolEncoding};

impl MaskKernel for ByteBoolEncoding {
    fn mask(&self, array: &ByteBoolArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(ByteBoolArray::new(array.buffer().clone(), array.validity().mask(mask)?).into_array())
    }
}

register_kernel!(MaskKernelAdapter(ByteBoolEncoding).lift());

impl TakeKernel for ByteBoolEncoding {
    fn take(&self, array: &ByteBoolArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let validity = array.validity_mask()?;
        let indices = indices.to_primitive()?;
        let bools = array.as_slice();

        // FIXME(ngates): we should be operating over canonical validity, which doesn't
        //  have fallible is_valid function.
        let arr = match validity {
            Mask::AllTrue(_) => {
                let bools = match_each_integer_ptype!(indices.ptype(), |$I| {
                    indices.as_slice::<$I>()
                    .iter()
                    .map(|&idx| {
                        let idx: usize = idx.as_();
                        bools[idx]
                    })
                    .collect::<Vec<_>>()
                });

                ByteBoolArray::from(bools).into_array()
            }
            Mask::AllFalse(_) => ByteBoolArray::from(vec![None; indices.len()]).into_array(),
            Mask::Values(values) => {
                let bools = match_each_integer_ptype!(indices.ptype(), |$I| {
                    indices.as_slice::<$I>()
                    .iter()
                    .map(|&idx| {
                        let idx = idx.as_();
                        if values.value(idx) {
                            Some(bools[idx])
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<Option<_>>>()
                });

                ByteBoolArray::from(bools).into_array()
            }
        };

        Ok(arr)
    }
}

register_kernel!(TakeKernelAdapter(ByteBoolEncoding).lift());

#[cfg(test)]
mod tests {
    use vortex_array::compute::conformance::mask::test_mask;
    use vortex_array::compute::{Operator, compare};

    use super::*;

    #[test]
    fn test_slice() {
        let original = vec![Some(true), Some(true), None, Some(false), None];
        let vortex_arr = ByteBoolArray::from(original);

        let sliced_arr = vortex_arr.slice(1, 4).unwrap();
        let sliced_arr = ByteBoolArray::try_from(sliced_arr).unwrap();

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

        let arr = compare(&lhs, &rhs, Operator::Eq).unwrap();

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

        let arr = compare(&lhs, &rhs, Operator::Eq).unwrap();

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

        let arr = compare(&lhs, &rhs, Operator::Eq).unwrap();

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
        test_mask(&ByteBoolArray::from(vec![true, false, true, true, false]));
        test_mask(&ByteBoolArray::from(vec![
            Some(true),
            Some(true),
            None,
            Some(false),
            None,
        ]));
    }
}

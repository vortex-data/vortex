// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod filter;
mod mask;
pub(crate) mod rules;
mod slice;
mod take;

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::NullArray;
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::DType;

    #[test]
    fn test_slice_nulls() {
        let nulls = NullArray::new(10);
        let sliced = nulls.slice(0..4).unwrap().to_null();

        assert_eq!(sliced.len(), 4);
        assert!(matches!(
            sliced.as_array().validity_mask().unwrap(),
            Mask::AllFalse(4)
        ));
    }

    #[test]
    fn test_take_nulls() {
        let nulls = NullArray::new(10);
        let taken = nulls
            .take(buffer![0u64, 2, 4, 6, 8].into_array())
            .unwrap()
            .to_null();

        assert_eq!(taken.len(), 5);
        assert!(matches!(
            taken.as_array().validity_mask().unwrap(),
            Mask::AllFalse(5)
        ));
    }

    #[test]
    fn test_scalar_at_nulls() {
        let nulls = NullArray::new(10);

        let scalar = nulls.scalar_at(0).unwrap();
        assert!(scalar.is_null());
        assert_eq!(scalar.dtype().clone(), DType::Null);
    }

    #[test]
    fn test_filter_null_array() {
        test_filter_conformance(&NullArray::new(5).into_array());
        test_filter_conformance(&NullArray::new(1).into_array());
        test_filter_conformance(&NullArray::new(10).into_array());
    }

    #[test]
    fn test_mask_null_array() {
        test_mask_conformance(&NullArray::new(5).into_array());
    }

    #[test]
    fn test_take_null_array_conformance() {
        test_take_conformance(&NullArray::new(5).into_array());
        test_take_conformance(&NullArray::new(1).into_array());
        test_take_conformance(&NullArray::new(10).into_array());
    }

    #[rstest]
    // From test_all_consistency
    #[case::null_array_small(NullArray::new(5))]
    #[case::null_array_medium(NullArray::new(100))]
    // Additional test cases
    #[case::null_array_single(NullArray::new(1))]
    #[case::null_array_large(NullArray::new(1000))]
    #[case::null_array_empty(NullArray::new(0))]
    fn test_null_consistency(#[case] array: NullArray) {
        test_array_consistency(&array.into_array());
    }
}

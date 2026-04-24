// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_mask::MaskValues;

use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::filter::execute::bitbuffer;
use crate::arrays::filter::execute::filter_validity;

pub fn filter_bool(array: &BoolArray, mask: &Arc<MaskValues>) -> BoolArray {
    let validity = array
        .validity()
        .vortex_expect("bool validity should be derivable");
    let filtered_validity = filter_validity(validity, mask);

    let bit_buffer = array.to_bit_buffer();
    let filtered_buffer = bitbuffer::filter_bit_buffer(&bit_buffer, mask.as_ref());

    BoolArray::new(filtered_buffer, filtered_validity)
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use rstest::rstest;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::arrays::filter::execute::bool::BoolArray;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
    use crate::compute::conformance::filter::test_filter_conformance;

    #[test]
    fn filter_bool_test() {
        let arr = BoolArray::from_iter([true, true, false]);
        let mask = Mask::from_iter([true, false, true]);

        #[expect(deprecated)]
        let filtered = arr.filter(mask).unwrap().to_bool();
        assert_eq!(2, filtered.len());

        assert_eq!(
            vec![true, false],
            filtered.into_bit_buffer().iter().collect_vec()
        )
    }

    #[rstest]
    #[case(BoolArray::from_iter([true, false, true, true, false]))]
    #[case(BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]))]
    #[case(BoolArray::from_iter([true]))]
    #[case(BoolArray::from_iter([false, false]))]
    #[case(BoolArray::from_iter((0..100).map(|i| i % 2 == 0)))]
    #[case(BoolArray::from_iter((0..1024).map(|i| i % 3 != 0)))]
    fn test_filter_bool_conformance(#[case] array: BoolArray) {
        test_filter_conformance(&array.into_array());
    }
}

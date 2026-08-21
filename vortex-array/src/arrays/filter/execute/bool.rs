// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_mask::MaskValuesRef;

use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::filter::execute::bitbuffer;
use crate::arrays::filter::execute::filter_validity;

pub fn filter_bool(array: &BoolArray, mask: &MaskValuesRef) -> BoolArray {
    let validity = array
        .validity()
        .vortex_expect("validity is derivable for a valid BoolArray");
    let filtered_validity = filter_validity(validity, mask);

    let bit_buffer = array.to_bit_buffer();
    let filtered_buffer = bitbuffer::filter_bit_buffer(&bit_buffer, mask);

    BoolArray::new(filtered_buffer, filtered_validity)
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rstest::rstest;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::filter::execute::bool::BoolArray;
    use crate::compute::conformance::filter::test_filter_conformance;

    #[test]
    fn filter_bool_test() {
        let mut ctx = array_session().create_execution_ctx();
        let arr = BoolArray::from_iter([true, true, false]);
        let mask = Mask::from_iter([true, false, true]);

        let filtered = arr
            .filter(mask)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
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
        test_filter_conformance(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}

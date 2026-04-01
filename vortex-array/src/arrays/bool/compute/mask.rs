// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for Bool {
    fn mask(array: ArrayView<'_, Bool>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            BoolArray::new(
                array.to_bit_buffer(),
                array.validity().and(Validity::Array(mask.clone()))?,
            )
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::compute::conformance::mask::test_mask_conformance;

    #[rstest]
    #[case(BoolArray::from_iter([true, false, true, true, false]))]
    #[case(BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]))]
    #[case(BoolArray::from_iter([true]))]
    #[case(BoolArray::from_iter([false, false]))]
    #[case(BoolArray::from_iter((0..100).map(|i| i % 2 == 0)))]
    fn test_mask_bool_conformance(#[case] array: BoolArray) {
        test_mask_conformance(&array.into_array());
    }
}

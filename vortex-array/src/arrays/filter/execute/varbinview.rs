// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::BooleanArray;
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::filter::execute::values_to_mask;
use crate::arrow::FromArrowArray;
use crate::arrow::IntoArrowArray;

pub fn filter_varbinview(array: &VarBinViewArray, mask: &Arc<MaskValues>) -> VarBinViewArray {
    // Delegate to the Arrow implementation of filter over `VarBinView`.
    arrow_filter_fn(&array.clone().into_array(), &values_to_mask(mask))
        .vortex_expect("VarBinViewArray is Arrow-compatible and supports arrow_filter_fn")
        .as_::<VarBinView>()
        .into_owned()
}

fn arrow_filter_fn(array: &ArrayRef, mask: &Mask) -> vortex_error::VortexResult<ArrayRef> {
    let values = match &mask {
        Mask::Values(values) => values,
        Mask::AllTrue(_) | Mask::AllFalse(_) => unreachable!("check in filter invoke"),
    };

    let array_ref = array.clone().into_arrow_preferred()?;
    let mask_array = BooleanArray::new(values.bit_buffer().clone().into(), None);
    let filtered = arrow_select::filter::filter(array_ref.as_ref(), &mask_array)?;

    ArrayRef::from_arrow(filtered.as_ref(), array.dtype().is_nullable())
}

#[cfg(test)]
mod test {
    use crate::IntoArray;
    use crate::arrays::VarBinViewArray;
    use crate::compute::conformance::filter::test_filter_conformance;

    #[test]
    fn test_filter_varbinview_conformance() {
        test_filter_conformance(
            &VarBinViewArray::from_iter_str(["one", "two", "three", "four", "five"]).into_array(),
        );

        test_filter_conformance(
            &VarBinViewArray::from_iter_nullable_str([
                Some("one"),
                None,
                Some("three"),
                Some("four"),
                Some("five"),
            ])
            .into_array(),
        );
    }
}

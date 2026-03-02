// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_mask::MaskValues;

use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::arrays::filter::execute::values_to_mask;
use crate::compute::arrow_filter_fn;

pub fn filter_varbinview(array: &VarBinViewArray, mask: &Arc<MaskValues>) -> VarBinViewArray {
    // Delegate to the Arrow implementation of filter over `VarBinView`.
    arrow_filter_fn(&array.to_array(), &values_to_mask(mask))
        .vortex_expect("VarBinViewArray is Arrow-compatible and supports arrow_filter_fn")
        .as_::<VarBinViewVTable>()
        .clone()
}

#[cfg(test)]
mod test {
    use crate::arrays::VarBinViewArray;
    use crate::compute::conformance::filter::test_filter_conformance;

    #[test]
    fn test_filter_varbinview_conformance() {
        test_filter_conformance(
            &VarBinViewArray::from_iter_str(["one", "two", "three", "four", "five"]).to_array(),
        );

        test_filter_conformance(
            &VarBinViewArray::from_iter_nullable_str([
                Some("one"),
                None,
                Some("three"),
                Some("four"),
                Some("five"),
            ])
            .to_array(),
        );
    }
}

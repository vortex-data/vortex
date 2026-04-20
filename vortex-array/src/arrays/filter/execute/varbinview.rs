// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_mask::MaskValues;

use crate::arrays::VarBinViewArray;
use crate::arrow_hooks::arrow_compute;

pub fn filter_varbinview(array: &VarBinViewArray, mask: &Arc<MaskValues>) -> VarBinViewArray {
    (arrow_compute().vortex_expect("arrow compute not registered").filter_varbinview)(array, mask)
        .vortex_expect("filter_varbinview via arrow compute")
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

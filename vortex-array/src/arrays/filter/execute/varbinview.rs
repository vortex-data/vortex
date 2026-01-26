// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_mask::Mask;

use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::compute::arrow_filter_fn;

pub fn filter_varbinview(array: &VarBinViewArray, mask: &Mask) -> VarBinViewArray {
    arrow_filter_fn(array.as_ref(), mask)
        .vortex_expect("filter varbinview array")
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
            VarBinViewArray::from_iter_str(["one", "two", "three", "four", "five"]).as_ref(),
        );

        test_filter_conformance(
            VarBinViewArray::from_iter_nullable_str([
                Some("one"),
                None,
                Some("three"),
                Some("four"),
                Some("five"),
            ])
            .as_ref(),
        );
    }
}

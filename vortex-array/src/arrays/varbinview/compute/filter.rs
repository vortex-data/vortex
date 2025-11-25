// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::arrays::VarBinViewVTable;
use crate::compute::FilterKernel;
use crate::compute::FilterKernelAdapter;
use crate::compute::arrow_filter_fn;
use crate::register_kernel;

impl FilterKernel for VarBinViewVTable {
    fn filter(&self, array: &Self::Array, mask: &Mask) -> VortexResult<ArrayRef> {
        arrow_filter_fn(array.as_ref(), mask)
    }
}

register_kernel!(FilterKernelAdapter(VarBinViewVTable).lift());

#[cfg(test)]
mod tests {
    use crate::arrays::VarBinViewArray;
    use crate::compute::conformance::filter::test_filter_conformance;

    #[test]
    fn test_filter_var_bin_view_array() {
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

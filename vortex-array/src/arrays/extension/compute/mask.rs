// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::compute::MaskKernel;
use crate::compute::MaskKernelAdapter;
use crate::compute::mask;
use crate::register_kernel;

impl MaskKernel for ExtensionVTable {
    fn mask(&self, array: &ExtensionArray, mask_array: &Mask) -> VortexResult<ArrayRef> {
        // Use compute::mask directly since mask_array has compute::mask semantics (true=null)
        let masked_storage = mask(array.storage(), mask_array)?;
        assert!(masked_storage.dtype().is_nullable());

        Ok(ExtensionArray::new(
            array
                .ext_dtype()
                .with_nullability(masked_storage.dtype().nullability()),
            masked_storage,
        )
        .into_array())
    }
}

register_kernel!(MaskKernelAdapter(ExtensionVTable).lift());

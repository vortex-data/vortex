// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::ExtDType;
use vortex_error::VortexResult;
use vortex_mask::Mask;

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
        if masked_storage.dtype().nullability() == array.ext_dtype().storage_dtype().nullability() {
            Ok(ExtensionArray::new(array.ext_dtype().clone(), masked_storage).into_array())
        } else {
            // The storage dtype changed (i.e., became nullable due to masking)
            let ext_dtype = Arc::new(ExtDType::new(
                array.ext_dtype().id().clone(),
                Arc::new(masked_storage.dtype().clone()),
                array.ext_dtype().metadata().cloned(),
            ));
            Ok(ExtensionArray::new(ext_dtype, masked_storage).into_array())
        }
    }
}

register_kernel!(MaskKernelAdapter(ExtensionVTable).lift());

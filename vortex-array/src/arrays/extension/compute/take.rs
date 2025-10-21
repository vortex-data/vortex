// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::ExtDType;
use vortex_error::VortexResult;

use crate::arrays::{ExtensionArray, ExtensionVTable};
use crate::compute::{self, TakeKernel, TakeKernelAdapter};
use crate::{Array, ArrayRef, IntoArray, register_kernel};

impl TakeKernel for ExtensionVTable {
    fn take(&self, array: &ExtensionArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let taken_storage = compute::take(array.storage(), indices)?;
        if taken_storage.dtype().nullability() == array.ext_dtype().storage_dtype().nullability() {
            Ok(ExtensionArray::new(array.ext_dtype().clone(), taken_storage).into_array())
        } else {
            // The storage dtype changed (i.e., became nullable due to nullable indices)
            let ext_dtype = Arc::new(ExtDType::new(
                array.ext_dtype().id().clone(),
                Arc::new(taken_storage.dtype().clone()),
                array.ext_dtype().metadata().cloned(),
            ));
            Ok(ExtensionArray::new(ext_dtype, taken_storage).into_array())
        }
    }
}

register_kernel!(TakeKernelAdapter(ExtensionVTable).lift());

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::compute::TakeKernel;
use crate::compute::TakeKernelAdapter;
use crate::compute::{self};
use crate::register_kernel;

impl TakeKernel for ExtensionVTable {
    fn take(&self, array: &ExtensionArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let taken_storage = compute::take(array.storage(), indices)?;
        Ok(ExtensionArray::new(
            array
                .ext_dtype()
                .with_nullability(taken_storage.dtype().nullability()),
            taken_storage,
        )
        .into_array())
    }
}

register_kernel!(TakeKernelAdapter(ExtensionVTable).lift());

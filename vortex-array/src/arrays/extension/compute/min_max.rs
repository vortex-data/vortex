// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::compute::{self};
use crate::register_kernel;

impl MinMaxKernel for ExtensionVTable {
    fn min_max(&self, array: &ExtensionArray) -> VortexResult<Option<MinMaxResult>> {
        let non_nullable_ext_dtype = array.ext_dtype().with_nullability(Nullability::NonNullable);
        Ok(
            compute::min_max(array.storage())?.map(|MinMaxResult { min, max }| MinMaxResult {
                min: Scalar::extension_ref(non_nullable_ext_dtype.clone(), min),
                max: Scalar::extension_ref(non_nullable_ext_dtype, max),
            }),
        )
    }
}

register_kernel!(MinMaxKernelAdapter(ExtensionVTable).lift());

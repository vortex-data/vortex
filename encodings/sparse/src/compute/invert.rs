// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::compute::InvertKernel;
use vortex_array::compute::InvertKernelAdapter;
use vortex_array::compute::invert;
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::SparseArray;
use crate::SparseVTable;

impl InvertKernel for SparseVTable {
    fn invert(&self, array: &SparseArray) -> VortexResult<ArrayRef> {
        let inverted_fill = array.fill_scalar().as_bool().invert().into_scalar();
        let inverted_patches = array
            .patches()
            .clone()
            .map_values(|values| invert(&values))?;
        SparseArray::try_new_from_patches(inverted_patches, inverted_fill).map(|a| a.into_array())
    }
}

register_kernel!(InvertKernelAdapter(SparseVTable).lift());

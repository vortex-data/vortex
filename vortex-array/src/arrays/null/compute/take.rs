// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::NullArray;
use crate::arrays::NullVTable;
use crate::compute::TakeKernel;
use crate::compute::TakeKernelAdapter;
use crate::register_kernel;

impl TakeKernel for NullVTable {
    #[allow(clippy::cast_possible_truncation)]
    fn take(&self, array: &NullArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = indices.to_primitive()?;

        // Enforce all indices are valid
        match_each_integer_ptype!(indices.ptype(), |T| {
            for index in indices.as_slice::<T>() {
                if (*index as usize) >= array.len() {
                    vortex_bail!(OutOfBounds: *index as usize, 0, array.len());
                }
            }
        });

        Ok(NullArray::new(indices.len()).into_array())
    }
}

register_kernel!(TakeKernelAdapter(NullVTable).lift());

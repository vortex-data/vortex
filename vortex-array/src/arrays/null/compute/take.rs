// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::match_each_integer_ptype;
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::{NullArray, NullVTable};
use crate::compute::{TakeKernel, TakeKernelAdapter};
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

impl TakeKernel for NullVTable {
    #[allow(clippy::cast_possible_truncation)]
    fn take(&self, array: &NullArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = indices.to_primitive();

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

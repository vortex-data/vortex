// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::compute::InvertKernel;
use crate::compute::InvertKernelAdapter;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

impl InvertKernel for BoolVTable {
    fn invert(&self, array: &BoolArray) -> VortexResult<ArrayRef> {
        Ok(BoolArray::new(array.to_bit_buffer().not(), array.validity().clone()).into_array())
    }
}

register_kernel!(InvertKernelAdapter(BoolVTable).lift());

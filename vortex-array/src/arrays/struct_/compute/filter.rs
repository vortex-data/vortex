// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::compute::FilterKernel;
use crate::compute::FilterKernelAdapter;
use crate::compute::filter;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

impl FilterKernel for StructVTable {
    fn filter(&self, array: &StructArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let validity = array.validity().filter(mask)?;

        let fields: Vec<ArrayRef> = array
            .unmasked_fields()
            .iter()
            .map(|field| filter(field, mask))
            .try_collect()?;
        let length = fields
            .first()
            .map(|a| a.len())
            .unwrap_or_else(|| mask.true_count());

        StructArray::try_new_with_dtype(fields, array.struct_fields().clone(), length, validity)
            .map(|a| a.into_array())
    }
}

register_kernel!(FilterKernelAdapter(StructVTable).lift());

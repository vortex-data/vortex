// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::DictArray;
use crate::arrays::DictVTable;
use crate::arrays::SliceKernel;

impl SliceKernel for DictVTable {
    fn slice(
        array: &Self::Array,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let sliced_code = array.codes().slice(range)?;
        if sliced_code.is::<ConstantVTable>() {
            let code = &sliced_code.scalar_at(0)?.as_primitive().as_::<usize>();
            return if let Some(code) = code {
                Ok(Some(
                    ConstantArray::new(array.values().scalar_at(*code)?, sliced_code.len())
                        .into_array(),
                ))
            } else {
                Ok(Some(
                    ConstantArray::new(Scalar::null(array.dtype().clone()), sliced_code.len())
                        .to_array(),
                ))
            };
        }
        // SAFETY: slicing the codes preserves invariants.
        Ok(Some(
            unsafe { DictArray::new_unchecked(sliced_code, array.values().clone()) }.into_array(),
        ))
    }
}

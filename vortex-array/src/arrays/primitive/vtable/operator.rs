// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_compute::filter::Filter;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_vector::PVector;

use crate::ArrayRef;
use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::vtable::{OperatorVTable, ValidityHelper};

impl OperatorVTable<PrimitiveVTable> for PrimitiveVTable {
    fn bind(
        array: &PrimitiveArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let mask = ctx.bind_selection(array.len(), selection)?;
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;

        match_each_native_ptype!(array.ptype(), |T| {
            let elements = array.buffer::<T>();
            Ok(kernel(move || {
                let mask = mask.execute()?;
                let validity = validity.execute()?;

                // Note that validity already has the mask applied so we only need to apply it to
                // the elements.
                let elements = elements.filter(&mask);

                Ok(PVector::try_new(elements, validity)?.into())
            }))
        })
    }
}

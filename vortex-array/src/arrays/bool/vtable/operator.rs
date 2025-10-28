// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_compute::filter::Filter;
use vortex_error::VortexResult;
use vortex_vector::BoolVector;

use crate::ArrayRef;
use crate::arrays::{BoolArray, BoolVTable};
use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::vtable::{OperatorVTable, ValidityHelper};

impl OperatorVTable<BoolVTable> for BoolVTable {
    fn bind(
        array: &BoolArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let bits = array.buffer.clone();
        let mask = ctx.bind_selection(array.len(), selection)?;
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;

        Ok(kernel(move || {
            let mask = mask.execute()?;
            let validity = validity.execute()?;

            // Note that validity already has the mask applied so we only need to apply it to bits.
            let bits = bits.filter(&mask);

            Ok(BoolVector::try_new(bits, validity)?.into())
        }))
    }
}

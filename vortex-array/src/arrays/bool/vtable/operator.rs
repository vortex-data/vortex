// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::try_join;
use vortex_compute::filter::Filter;
use vortex_error::VortexResult;
use vortex_vector::BoolVector;

use crate::ArrayRef;
use crate::arrays::{BoolArray, BoolVTable};
use crate::execution::{BatchKernel, BindCtx, kernel};
use crate::vtable::{OperatorVTable, ValidityHelper};

impl OperatorVTable<BoolVTable> for BoolVTable {
    fn bind(
        array: &BoolArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<Box<dyn BatchKernel>> {
        let bits = array.buffer.clone();
        let mask = ctx.bind_selection(array.len(), selection)?;
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;

        Ok(kernel(|_out| async move {
            let (mask, validity) = try_join!(mask.execute(), validity.execute())?;

            // Note that validity already has the mask applied so we only need to apply it to bits.
            let bits = bits.filter(&mask);

            Ok(BoolVector::new(bits, validity).into())
        }))
    }
}

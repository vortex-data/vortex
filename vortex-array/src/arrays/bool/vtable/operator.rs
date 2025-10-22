// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::{BoolArray, BoolVTable};
use crate::compute::vectors::filter::Filter;
use crate::execution::{kernel, BatchKernel, BindCtx};
use crate::vtable::{OperatorVTable, ValidityHelper};
use crate::ArrayRef;
use futures::try_join;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_vector::BoolVector;

impl OperatorVTable<BoolVTable> for BoolVTable {
    fn bind(
        array: &BoolArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<Box<dyn BatchKernel>> {
        let bits = BitBuffer::from(array.buffer.clone());
        let mask = ctx.bind_selection(array.len(), selection)?;
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;

        Ok(kernel(|out| async move {
            let (mask, validity) = try_join!(mask.execute(), validity.execute())?;

            // Note that validity already has the mask applied so we only need to apply it to bits.
            let (bits_out, _) = out.into_bool().into_parts();
            let bits = bits.filter_into(&mask, bits_out);

            Ok(BoolVector::new(bits, validity).into())
        }))
    }
}

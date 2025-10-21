// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::{BoolArray, BoolVTable};
use crate::execution::{kernel, BatchKernel, BindCtx};
use crate::vtable::{OperatorVTable, ValidityHelper};
use crate::ArrayRef;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

impl OperatorVTable<BoolVTable> for BoolVTable {
    fn bind(
        array: &BoolArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<Box<dyn BatchKernel>> {
        let bits = BitBuffer::from(array.buffer.clone());

        let validity = ctx.bind_validity(array.validity(), selection, ctx)?;

        Ok(kernel(|out| async move {}))
    }
}

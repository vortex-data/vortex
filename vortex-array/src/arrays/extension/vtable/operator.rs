// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::execution::BatchKernelRef;
use crate::execution::BindCtx;
use crate::vtable::OperatorVTable;

impl OperatorVTable<ExtensionVTable> for ExtensionVTable {
    fn bind(
        array: &ExtensionArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        // Since vectors are physically typed, extension array can delegate to its storage array
        ctx.bind(&array.storage, selection)
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::arrays::TakeExecute;
use crate::arrays::TakeExecuteAdaptor;
use crate::compute::{self};
use crate::kernel::ParentKernelSet;

fn take_extension(array: &ExtensionArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
    let taken_storage = compute::take(array.storage(), indices)?;
    Ok(ExtensionArray::new(
        array
            .ext_dtype()
            .with_nullability(taken_storage.dtype().nullability()),
        taken_storage,
    )
    .into_array())
}

impl TakeExecute for ExtensionVTable {
    fn take(
        array: &ExtensionArray,
        indices: &dyn Array,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        take_extension(array, indices).map(Some)
    }
}

impl ExtensionVTable {
    pub const TAKE_KERNELS: ParentKernelSet<Self> =
        ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor::<Self>(Self))]);
}

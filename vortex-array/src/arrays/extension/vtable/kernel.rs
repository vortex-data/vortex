// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ArrowExport;
use crate::arrays::ArrowExportArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::NativeArrowArray;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ExecuteParentKernel;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<Extension> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&ExtensionArrowExportKernel),
    ParentKernelSet::lift(&CompareExecuteAdaptor(Extension)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(Extension)),
]);

#[derive(Debug)]
struct ExtensionArrowExportKernel;

impl ExecuteParentKernel<Extension> for ExtensionArrowExportKernel {
    type Parent = ArrowExport;

    fn execute_parent(
        &self,
        array: &ExtensionArray,
        parent: &ArrowExportArray,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let ext_dtype = array.ext_dtype().clone();
        let storage = array.storage_array().clone();
        let arrow = ext_dtype.to_arrow_array(storage, parent.target(), ctx)?;
        Ok(Some(
            NativeArrowArray::new(arrow, parent.dtype().clone()).into_array(),
        ))
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_schema::DataType;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ArrowExport;
use crate::arrays::ArrowExportArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::NativeArrowArray;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::executor::dictionary::make_dict_array;
use crate::kernel::ExecuteParentKernel;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<Dict> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(Dict)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(Dict)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(Dict)),
    ParentKernelSet::lift(&DictArrowExportKernel),
]);

#[derive(Debug)]
struct DictArrowExportKernel;

impl ExecuteParentKernel<Dict> for DictArrowExportKernel {
    type Parent = ArrowExport;

    fn execute_parent(
        &self,
        array: &DictArray,
        parent: &ArrowExportArray,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let DataType::Dictionary(codes_type, values_type) = parent.target() else {
            return Ok(None);
        };
        let parts = array.clone().into_parts();
        let codes = parts.codes.execute_arrow(Some(codes_type), ctx)?;
        let values = parts.values.execute_arrow(Some(values_type), ctx)?;
        let arrow = make_dict_array(codes_type, codes, values)?;
        Ok(Some(
            NativeArrowArray::new(arrow, parent.dtype().clone()).into_array(),
        ))
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_schema::DataType;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ArrowExport;
use crate::arrays::ArrowExportArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::NativeArrowArray;
use crate::kernel::ExecuteParentKernel;
use crate::kernel::ParentKernelSet;

pub(super) const PARENT_KERNELS: ParentKernelSet<Constant> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&ConstantArrowExportKernel)]);

#[derive(Debug)]
struct ConstantArrowExportKernel;

impl ExecuteParentKernel<Constant> for ConstantArrowExportKernel {
    type Parent = ArrowExport;

    fn execute_parent(
        &self,
        array: &ConstantArray,
        parent: &ArrowExportArray,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        match parent.target() {
            DataType::Dictionary(codes_type, values_type) => {
                let arrow = super::constant_to_dict(array, codes_type, values_type, ctx)?;
                Ok(Some(
                    NativeArrowArray::new(arrow, parent.dtype().clone()).into_array(),
                ))
            }
            DataType::RunEndEncoded(ends_field, values_field) => {
                let arrow =
                    super::constant_to_run_end(array, ends_field.data_type(), values_field, ctx)?;
                Ok(Some(
                    NativeArrowArray::new(arrow, parent.dtype().clone()).into_array(),
                ))
            }
            _ => Ok(None),
        }
    }
}

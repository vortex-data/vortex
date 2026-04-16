// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Variant;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::variant::VariantArrayExt;
use crate::kernel::ExecuteParentKernel;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::variant_get::VariantGet;

pub(super) const PARENT_KERNELS: ParentKernelSet<Variant> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&VariantGetExecuteParent)]);

#[derive(Debug)]
struct VariantGetExecuteParent;

impl ExecuteParentKernel<Variant> for VariantGetExecuteParent {
    type Parent = ExactScalarFn<VariantGet>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Variant>,
        parent: ScalarFnArrayView<'_, VariantGet>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        assert_eq!(child_idx, 0);

        let core_storage = array.core_storage();
        if core_storage.is::<Variant>() {
            return Ok(None);
        }

        core_storage.execute_parent(&parent, child_idx, ctx)
    }
}

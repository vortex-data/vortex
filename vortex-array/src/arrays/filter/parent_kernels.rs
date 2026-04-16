// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parent kernels for wrapper-side execution over [`Filter`].
//!
//! [`VariantGet`](crate::scalar_fn::fns::variant_get::VariantGet) cannot execute directly. When a
//! `Filter` wrapper sits between the scalar function and the underlying variant encoding, the
//! wrapper has to preserve the whole variant-typed child through filtering and then re-dispatch the
//! scalar function on the filtered result.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Filter;
use crate::arrays::filter::FilterArrayExt;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::dtype::DType;
use crate::kernel::ExecuteParentKernel;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::variant_get::VariantGet;

pub(super) const PARENT_KERNELS: ParentKernelSet<Filter> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&FilterVariantGetExecuteParent)]);

#[derive(Debug)]
struct FilterVariantGetExecuteParent;

impl ExecuteParentKernel<Filter> for FilterVariantGetExecuteParent {
    type Parent = ExactScalarFn<VariantGet>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Filter>,
        parent: ScalarFnArrayView<'_, VariantGet>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }

        let child = array.child();
        if !matches!(child.dtype(), DType::Variant(_)) {
            return Ok(None);
        }

        let filtered = child
            .filter(array.filter_mask().clone())?
            .execute::<ArrayRef>(ctx)?;

        VariantGet
            .try_new_array(parent.len(), parent.options.clone(), [filtered])?
            .execute::<ArrayRef>(ctx)
            .map(Some)
    }
}

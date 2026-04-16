// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parent kernels for wrapper-side execution over [`Slice`].
//!
//! [`VariantGet`](crate::scalar_fn::fns::variant_get::VariantGet) cannot execute directly. When a
//! `Slice` wrapper sits between the scalar function and the underlying variant encoding, the
//! wrapper has to preserve the whole variant-typed child through slicing and then re-dispatch the
//! scalar function on the sliced result.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Slice;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::arrays::slice::SliceArrayExt;
use crate::dtype::DType;
use crate::kernel::ExecuteParentKernel;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::variant_get::VariantGet;

pub(super) const PARENT_KERNELS: ParentKernelSet<Slice> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&SliceVariantGetExecuteParent)]);

#[derive(Debug)]
struct SliceVariantGetExecuteParent;

impl ExecuteParentKernel<Slice> for SliceVariantGetExecuteParent {
    type Parent = ExactScalarFn<VariantGet>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Slice>,
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

        let sliced = child
            .slice(array.slice_range().clone())?
            .execute::<ArrayRef>(ctx)?;

        VariantGet
            .try_new_array(parent.len(), parent.options.clone(), [sliced])?
            .execute::<ArrayRef>(ctx)
            .map(Some)
    }
}

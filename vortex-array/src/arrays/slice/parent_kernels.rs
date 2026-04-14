// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parent kernels for wrapper-side execution over [`Slice`].
//!
//! Most scalar functions can commute with `Slice` via generic scalar-fn rules, but
//! [`VariantGet`](crate::scalar_fn::fns::variant_get::VariantGet) is unusual: it has to reach the
//! underlying variant encoding's `execute_parent` kernel rather than executing directly.
//!
//! This module keeps that wrapper-specific pass-through on the wrapper side. When a
//! `VariantGet` parent sits directly above a `Slice`, we slice the wrapped variant child first and
//! then re-dispatch `VariantGet` on the sliced child so execution can continue at the underlying
//! variant encoding.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Slice;
use crate::arrays::Variant;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::arrays::slice::SliceArrayExt;
use crate::arrays::variant::VariantArrayExt;
use crate::dtype::DType;
use crate::kernel::ExecuteParentKernel;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::variant_get::VariantGet;

pub(super) static PARENT_KERNELS: ParentKernelSet<Slice> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&SliceVariantGetExecuteParent)]);

/// Pass `variant_get` through a `Slice` wrapper to the underlying variant child.
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
        let inner = if let Some(variant) = child.as_opt::<Variant>() {
            variant.child().clone()
        } else if matches!(child.dtype(), DType::Variant(_)) {
            child.clone()
        } else {
            return Ok(None);
        };

        let sliced = inner
            .slice(array.slice_range().clone())?
            .execute::<ArrayRef>(ctx)?;

        VariantGet
            .try_new_array(parent.len(), parent.options.clone(), [sliced])?
            .execute::<ArrayRef>(ctx)
            .map(Some)
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parent kernels for wrapper-side execution over [`Filter`].
//!
//! [`VariantGet`](crate::scalar_fn::fns::variant_get::VariantGet) relies on the underlying
//! variant encoding to perform the real extraction work, so it cannot execute directly once a
//! `Filter` wrapper sits between the expression and that encoding.
//!
//! This module handles that pass-through at the wrapper layer by filtering the wrapped variant
//! child first and then re-dispatching `VariantGet` on the filtered child.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Filter;
use crate::arrays::Variant;
use crate::arrays::filter::FilterArrayExt;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::arrays::variant::VariantArrayExt;
use crate::dtype::DType;
use crate::kernel::ExecuteParentKernel;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::variant_get::VariantGet;

pub(super) static PARENT_KERNELS: ParentKernelSet<Filter> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&FilterVariantGetExecuteParent)]);

/// Pass `variant_get` through a `Filter` wrapper to the underlying variant child.
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
        let inner = if let Some(variant) = child.as_opt::<Variant>() {
            variant.child().clone()
        } else if matches!(child.dtype(), DType::Variant(_)) {
            child.clone()
        } else {
            return Ok(None);
        };

        let filtered = inner
            .filter(array.filter_mask().clone())?
            .execute::<ArrayRef>(ctx)?;

        VariantGet
            .try_new_array(parent.len(), parent.options.clone(), [filtered])?
            .execute::<ArrayRef>(ctx)
            .map(Some)
    }
}

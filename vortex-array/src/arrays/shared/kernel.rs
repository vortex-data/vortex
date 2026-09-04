// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Shared;
use crate::arrays::shared::current_array_ref_for_dispatch;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::like::Like;
use crate::scalar_fn::fns::like::LikeExecuteAdaptor;
use crate::scalar_fn::fns::like::LikeKernel;
use crate::scalar_fn::fns::like::LikeOptions;

impl LikeKernel for Shared {
    fn like(
        array: ArrayView<'_, Self>,
        pattern: &ArrayRef,
        options: LikeOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mut current = current_array_ref_for_dispatch(array)?.clone();
        while let Some(shared) = current.as_opt::<Shared>() {
            current = current_array_ref_for_dispatch(shared)?.clone();
        }

        // Give the current encoding's LIKE kernel a chance without removing Shared. If it
        // declines, normal execution will materialize Shared and retain that result for reuse.
        let parent = Like::try_new(current.clone(), pattern.clone(), options)?.into_array();

        ctx.try_execute_parent_kernel(&parent, &current, 0)
    }
}

pub(crate) fn initialize(session: &VortexSession) {
    session
        .kernels()
        .register_execute_parent_kernel(Like.id(), Shared, LikeExecuteAdaptor(Shared));
}

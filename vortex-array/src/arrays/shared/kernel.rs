// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ArrayVTable;
use crate::ExecutionCtx;
use crate::arrays::Shared;
use crate::arrays::shared::SharedArrayExt;
use crate::executor::execute_parent_for_child;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::kernels::ExecuteParentFn;

pub(crate) fn initialize(session: &VortexSession) {
    session
        .kernels()
        .register_execute_parent_for_any_parent(Shared.id(), &[execute_parent as ExecuteParentFn]);
}

fn execute_parent(
    child: &ArrayRef,
    parent: &ArrayRef,
    child_idx: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some(shared) = child.as_opt::<Shared>() else {
        return Ok(None);
    };

    let mut current = shared.current_array_ref().clone();
    while let Some(source) = current
        .as_opt::<Shared>()
        .map(|shared| shared.current_array_ref().clone())
    {
        current = source;
    }

    let kernels = ctx.execute_parent_kernels();
    execute_parent_for_child(parent, &current, child_idx, kernels.as_ref(), ctx)
}

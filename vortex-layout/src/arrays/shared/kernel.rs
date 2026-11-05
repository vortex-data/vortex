// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VTable;
use vortex_array::arrays::Dict;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::arrays::shared::Shared;
use crate::arrays::shared::SharedArraySlotsExt;

/// The slot of a dictionary that holds its values.
const DICT_VALUES_SLOT: usize = 1;

/// Registers the execute-parent kernels of [`Shared`] with `session`.
pub fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent(Like.id(), Shared.id(), &[like_on_source]);
    kernels.register_execute_parent(Dict.id(), Shared.id(), &[take_from_source]);
}

/// Whether a dictionary with `codes_len` codes reads so few of its string `values` that taking
/// them from the encoded values costs less than a decode of all of them.
pub(crate) fn is_sparse_string_take(values: &ArrayRef, codes_len: usize) -> bool {
    (values.dtype().is_utf8() || values.dtype().is_binary())
        && values.len() >= 1024
        && codes_len.saturating_mul(16) < values.len()
}

/// Whether the session has an execute-parent kernel for `parent` over `child`.
fn has_kernel(parent: &ArrayRef, child: &ArrayRef, ctx: &ExecutionCtx) -> bool {
    ctx.session()
        .kernels()
        .has_execute_parent(parent.encoding_id(), child.encoding_id())
}

/// Moves a `LIKE` onto the source of a share that is not computed yet, if the source encoding
/// has its own `LIKE` kernel.
///
/// A share canonicalizes its source once for all of its users. A `LIKE` kernel on an encoding
/// such as FSST or OnPair matches the encoded strings without a decode, which is faster than the
/// decode alone. A later user that needs all of the values still decodes the share once, and a
/// user that needs few of them can take them from the encoded source.
fn like_on_source(
    child: &ArrayRef,
    parent: &ArrayRef,
    child_idx: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some(shared) = child.as_opt::<Shared>() else {
        return Ok(None);
    };
    if shared.data().is_computed() {
        return Ok(None);
    }

    let source = shared.source().clone();
    if !has_kernel(parent, &source, ctx) {
        return Ok(None);
    }

    // SAFETY: the source of a share holds the same logical values as the share.
    let parent = unsafe { parent.clone().with_slot(child_idx, source) }?;
    Ok(Some(parent))
}

/// Takes the values of a dictionary from the source of its shared values, if the share is not
/// computed yet, the dictionary reads few of the values, and the source encoding has its own take
/// kernel.
///
/// The scan shares the values of a dictionary, so that its batches decode them once. A filter
/// that runs after the scan chose the values can keep so few codes that a take from the encoded
/// values costs less than the decode.
fn take_from_source(
    child: &ArrayRef,
    parent: &ArrayRef,
    child_idx: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if child_idx != DICT_VALUES_SLOT || !is_sparse_string_take(child, parent.len()) {
        return Ok(None);
    }
    let Some(shared) = child.as_opt::<Shared>() else {
        return Ok(None);
    };
    if shared.data().is_computed() {
        return Ok(None);
    }

    let source = shared.source().clone();
    if !has_kernel(parent, &source, ctx) {
        return Ok(None);
    }

    // SAFETY: the source of a share holds the same logical values as the share.
    let parent = unsafe { parent.clone().with_slot(child_idx, source) }?;
    Ok(Some(parent))
}

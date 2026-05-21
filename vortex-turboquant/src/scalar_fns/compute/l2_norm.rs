// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `L2Norm` execute-parent kernel that intercepts `L2Norm(TQDecode(tq))` and returns the stored
//! per-row norms directly instead of decoding and recomputing.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::optimizer::kernels::ExecuteParentFn;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;
use vortex_tensor::scalar_fns::l2_norm::L2Norm;

use crate::TQDecode;
use crate::vector::storage::parse_storage;
use crate::vtable::TurboQuant;

/// Register the `L2Norm(TQDecode(_))` execute-parent kernel on the session.
pub(super) fn register(session: &VortexSession) {
    session.kernels().register_execute_parent(
        L2Norm.id(),
        TQDecode.id(),
        &[l2_norm_tq_decode_execute_parent as ExecuteParentFn],
    );
}

/// Intercepts `L2Norm(TQDecode(tq_arr))` and returns the stored TurboQuant `norms` field.
///
/// The kernel only fires when both the parent matches `ExactScalarFn<L2Norm>` and the child
/// matches `ExactScalarFn<TQDecode>`. Returns `Ok(None)` for any other shape so the canonical
/// `L2Norm` path runs unchanged.
//
// This is semantically correct because TurboQuant stores per-row inverse direction norms and
// `TQDecode` applies that correction before re-applying the original row norm. In other words,
// valid nonzero decoded rows preserve the stored L2 norm even though coordinates are lossy.
fn l2_norm_tq_decode_execute_parent(
    child: &ArrayRef,
    parent: &ArrayRef,
    _child_idx: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if !parent.is::<ExactScalarFn<L2Norm>>() {
        return Ok(None);
    }
    if !child.is::<ExactScalarFn<TQDecode>>() {
        return Ok(None);
    }

    let tq_array = child.as_::<ScalarFn>().child_at(0).clone();

    // Defensive: TQDecode's signature already guarantees this, but a misregistration or a
    // future TQDecode that takes a wrapped child should fall back to the canonical path.
    if tq_array
        .dtype()
        .as_extension_opt()
        .and_then(|d| d.metadata_opt::<TurboQuant>())
        .is_none()
    {
        return Ok(None);
    }

    let parsed = parse_storage(tq_array, ctx)?;
    let norms_validity = parsed.norms.validity()?;
    let norms = PrimitiveArray::from_buffer_handle(
        parsed.norms.buffer_handle().clone(),
        parsed.norms.ptype(),
        norms_validity.and(parsed.vector_validity)?,
    )
    .into_array();

    vortex_ensure_eq!(
        norms.dtype(),
        parent.dtype(),
        "TurboQuant norms field dtype must match L2Norm output dtype"
    );
    vortex_ensure_eq!(
        norms.len(),
        parent.len(),
        "TurboQuant norms field length must match L2Norm output length"
    );

    Ok(Some(norms))
}

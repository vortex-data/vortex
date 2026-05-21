// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `L2Norm` execute-parent kernel that intercepts `L2Norm(TQDecode(tq))` and returns the
//! stored per-row norms directly instead of decoding and recomputing.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::dtype::Nullability;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::optimizer::kernels::ExecuteParentFn;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;
use vortex_tensor::scalar_fns::l2_norm::L2Norm;

use crate::TQDecode;
use crate::vector::storage::parse_storage_norms_only;

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
/// Semantically valid because [`TQDecode`] rescales each lossy quantized direction in flight
/// to unit norm before re-applying the stored row norm, so decoded rows preserve the stored
/// L2 norm to floating-point precision. Returning the stored field directly avoids the
/// inverse SORF transform, the per-row reciprocal, and the dimension truncation that the
/// canonical `L2Norm(execute(TQDecode))` path would otherwise run. The kernel returns
/// `Ok(None)` for any non-matching parent / child pair so the canonical path runs unchanged.
///
/// The result's nullability is coerced to the parent's expected dtype because the stored
/// `norms` child's validity may be wider than the outer struct (a shape
/// [`parse_storage_norms_only`] accepts).
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
    let parsed = parse_storage_norms_only(tq_array, ctx)?;

    let norms_validity = match parent.dtype().nullability() {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => parsed.vector_validity,
    };
    let norms = PrimitiveArray::from_buffer_handle(
        parsed.norms.buffer_handle().clone(),
        parsed.norms.ptype(),
        norms_validity,
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

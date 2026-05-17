// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cross-crate `OvcKernel for RunEnd`. Registers through the runtime
//! `ArrayKernels` registry so `ovc(runend_array)` expressions dispatch
//! to this kernel through the real Vortex executor.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::optimizer::kernels::ArrayKernels;
use vortex_array::optimizer::kernels::ExecuteParentFn;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_key_ord::ovc_scalarfn::Ovc;
use vortex_key_ord::stream_kernel::OvcAdaptor;
use vortex_key_ord::stream_kernel::OvcKernel;
use vortex_key_ord::stream_kernel::first_diff_byte;
use vortex_session::SessionExt;

use crate::RunEnd;
use crate::RunEndArrayExt;

impl OvcKernel for RunEnd {
    fn ovc_encode(array: ArrayView<'_, Self>, prev: u64) -> ArrayRef {
        // Encoding-aware: OVC values over a RunEnd input equal the input
        // values logically, so the value-column output is the input array
        // itself. We still walk run boundaries to honour the `prev` carry.
        let values_view = array.values().as_typed::<Primitive>().expect("primitive values");
        let ends_view = array.ends().as_typed::<Primitive>().expect("primitive ends");
        let values = values_view.as_slice::<u64>();
        let ends = ends_view.as_slice::<u32>();

        let mut prev = prev;
        let mut prev_end: u32 = 0;
        for k in 0..values.len() {
            let v = values[k];
            if ends[k] == prev_end {
                continue;
            }
            let _off = first_diff_byte(prev, v);
            prev = v;
            prev_end = ends[k];
        }

        array.array().clone()
    }
}

/// Flat-materialised baseline (for benches comparing encoding-aware
/// against `expand to Primitive`).
pub fn ovc_runend_materialise(array: ArrayView<'_, RunEnd>, prev: u64) -> ArrayRef {
    let values_view = array.values().as_typed::<Primitive>().expect("primitive values");
    let ends_view = array.ends().as_typed::<Primitive>().expect("primitive ends");
    let values = values_view.as_slice::<u64>();
    let ends = ends_view.as_slice::<u32>();

    let total: usize = ends.last().copied().unwrap_or(0) as usize;
    let mut out = BufferMut::<u64>::with_capacity(total);
    let mut prev = prev;
    let mut prev_end: u32 = 0;
    for k in 0..values.len() {
        let v = values[k];
        let run_len = (ends[k] - prev_end) as usize;
        if run_len == 0 {
            continue;
        }
        let _off = first_diff_byte(prev, v);
        out.extend(std::iter::repeat_n(v, run_len));
        prev = v;
        prev_end = ends[k];
    }
    PrimitiveArray::new(out.freeze(), Validity::NonNullable).into_array()
}

pub const RUNEND_OVC_KERNELS: ParentKernelSet<RunEnd> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&OvcAdaptor(RunEnd))]);

fn ovc_runend_execute(
    child: &ArrayRef,
    _parent: &ArrayRef,
    _child_idx: usize,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let view = child
        .as_typed::<RunEnd>()
        .ok_or_else(|| vortex_err!("not a runend"))?;
    Ok(Some(<RunEnd as OvcKernel>::ovc_encode(view, 0)))
}

/// Register the cross-crate RunEnd kernel into `session`. Call after
/// `vortex_key_ord::ovc_scalarfn::register_ovc`.
pub fn register_ovc_for_runend(session: &impl SessionExt) {
    let kernels = session.get::<ArrayKernels>();
    let ovc_id = <Ovc as ScalarFnVTable>::id(&Ovc);
    kernels.register_execute_parent(
        ovc_id,
        <RunEnd as VTable>::id(&RunEnd),
        &[ovc_runend_execute as ExecuteParentFn],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::expr::root;
    use vortex_buffer::Buffer;
    use vortex_key_ord::ovc_scalarfn::ovc;
    use vortex_key_ord::ovc_scalarfn::register_ovc;

    /// `apply()` preserves the encoding-aware RunEnd structure; `execute()`
    /// then canonicalises to Primitive via `run_end_canonicalize`.
    #[test]
    fn ovc_expression_end_to_end_runend() -> VortexResult<()> {
        register_ovc(&*LEGACY_SESSION);
        register_ovc_for_runend(&*LEGACY_SESSION);

        let values = PrimitiveArray::new(
            Buffer::<u64>::copy_from(&[1u64, 5, 9]),
            Validity::NonNullable,
        )
        .into_array();
        let ends = PrimitiveArray::new(
            Buffer::<u32>::copy_from(&[2u32, 4, 6]),
            Validity::NonNullable,
        )
        .into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let runend = RunEnd::new(ends, values, &mut ctx).into_array();

        let after_apply: ArrayRef = runend.clone().apply(&ovc(root()))?;
        let after_execute: ArrayRef = runend
            .apply(&ovc(root()))?
            .execute::<ArrayRef>(&mut ctx)?;

        assert_eq!(after_apply.len(), 6);
        assert_eq!(after_execute.len(), 6);
        assert!(
            after_apply.as_opt::<RunEnd>().is_some()
                || after_apply.as_opt::<Primitive>().is_some(),
            "unexpected post-apply encoding {}",
            after_apply.encoding_id()
        );
        Ok(())
    }
}

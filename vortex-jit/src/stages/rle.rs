// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::emit::{ArgKey, EmitCtx, SigBuilder};
use crate::form::PType;
use crate::stage::{JitStage, Placement};

/// RLE expansion as a `PostLoop` extern call.
///
/// RLE is fundamentally divergent — output positions don't map 1:1 to input
/// positions because each run expands to `run_length` outputs. This breaks
/// the elementwise stage model of §10. Two reasonable approaches:
///
///   1. *In-block scalar fill*: each emit() processes one or a few runs and
///      memsets their value into the output. Complex pointer bookkeeping
///      because runs span block boundaries arbitrarily.
///   2. *PostLoop extern call*: hand the whole `(values, lengths, n_runs)`
///      bundle to a Rust helper that produces the canonical output buffer in
///      one pass.
///
/// v1 uses (2). This means RLE doesn't fuse with downstream stages — the
/// output buffer is the kernel's terminal — but it demonstrates that the
/// framework's extern hook (§10) accommodates non-elementwise ops cleanly.
/// A future v2 with in-block run dispatch would compose with downstream
/// stages at the cost of substantially more emit() complexity.
#[derive(Debug, Clone, Copy)]
pub struct RleExpandPostLoop {
    pub ptype: PType,
    pub helper_name: &'static str,
}

impl JitStage for RleExpandPostLoop {
    fn tag(&self) -> &'static str {
        "RleExpandPostLoop"
    }

    fn fingerprint(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(1 + self.helper_name.len());
        v.push(self.ptype as u8);
        v.extend_from_slice(self.helper_name.as_bytes());
        v
    }

    fn placement(&self) -> Placement {
        Placement::PostLoop
    }

    fn declare(&self, sig: &mut SigBuilder) {
        sig.request_arg(ArgKey::OutPtr);
        sig.request_arg(ArgKey::Named("rle_values"));
        sig.request_arg(ArgKey::Named("rle_lengths"));
        sig.request_arg(ArgKey::Named("rle_n_runs"));
        sig.request_extern(self.helper_name);
    }

    fn emit(&self, cx: &mut EmitCtx<'_, '_>) -> VortexResult<()> {
        let out_ptr = cx.runtime_arg(&ArgKey::OutPtr)?;
        let vals = cx.runtime_arg(&ArgKey::Named("rle_values"))?;
        let lens = cx.runtime_arg(&ArgKey::Named("rle_lengths"))?;
        let n_runs_ptr = cx.runtime_arg(&ArgKey::Named("rle_n_runs"))?;
        let n_runs = cx.load_lane(n_runs_ptr, 0, PType::I64);
        drop(cx.extern_call_by_name(
            self.helper_name,
            &[out_ptr, vals, lens, n_runs],
        ));
        Ok(())
    }
}
